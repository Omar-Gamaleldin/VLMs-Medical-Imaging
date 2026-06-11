import os
import sys
import json
import time
import random
import base64
import argparse
import requests

from io import BytesIO
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
#  Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def encode_image_from_bytes(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_grayscale(image):
    return image.mode in ["L", "I;16"]


def normalize_16bit_to_8bit(image):
    return image.point(lambda x: (x / 256)).convert("L")


def ensure_rgb(image):
    if image.mode == "I;16":
        return normalize_16bit_to_8bit(image).convert("RGB")
    elif is_grayscale(image) or image.mode == "RGBA":
        return image.convert("RGB")
    elif image.mode == "RGB":
        return image.copy()
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")


def get_clean_image(image_path):
    with Image.open(image_path) as img:
        rgb_image = ensure_rgb(img)
    return encode_image_from_bytes(rgb_image)


def get_qa(img_file_name, json_dir):
    with open(json_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)
    result = next((entry['question_answer']
                   for entry in data if entry['filename'] == img_file_name), None)
    return [{'question': e['question'], 'answer': e['answer']} for e in result]


# ──────────────────────────────────────────────────────────────────────────────
#  Batch inference via /v1/chat/completions/batch
# ──────────────────────────────────────────────────────────────────────────────

def send_batch(batch_messages: list, server_url: str, model_name: str,
               thinking_budget: int = 4096) -> list[dict]:
    """
    POST all conversations in one request to /v1/chat/completions/batch.
    batch_messages: list of conversations, each conversation is a list of messages.
    Returns a list of response dicts, one per conversation.
    """
    payload = {
        "model": model_name,
        "messages": batch_messages,   # list of conversations
        "max_tokens": 5200,
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
    }

    resp = requests.post(
        f"{server_url}/v1/chat/completions/batch",
        json=payload,
        timeout=600,   # 10 min — large batches with images can be slow
    )
    resp.raise_for_status()
    data = resp.json()

    # The endpoint returns one choice per conversation under data["choices"].
    # Wrap each back into a response-like dict so parse_output works uniformly.
    print(data["choices"])
    return [{"choice": [choice], "usage": data.get("usage", {})}
            for choice in data["choices"]]


def parse_output(response: dict) -> tuple[str, str | None]:
    """Extract (text, reasoning_content) from a response dict."""
    message = response["choice"][0]["message"]
    text = message.get("content", "") or ""
    reasoning = message.get("reasoning_content", None)
    return text, reasoning


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5 Thinking — batch server inference")
    parser.add_argument("--data_dir",        type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--results_dir",     type=str,
                        default="control/results/qwen3.5_thinking",
                        help="Directory to write result JSON files")
    parser.add_argument("--server_url",      type=str,
                        default="http://localhost:8001",
                        help="vLLM server base URL (default: http://localhost:8001)")
    parser.add_argument("--model_name",      type=str,
                        default="Qwen3.5-9B",
                        help="Served model name — must match --served-model-name on the server")
    parser.add_argument("--experiments",     type=str, nargs="+",
                        default=["RQ1"], choices=["RQ1", "RQ2", "RQ3", "AS"],
                        help="Which experiments to run (default: RQ1)")
    parser.add_argument("--n_images",        type=int, default=500,
                        help="Number of images to sample per experiment (default: 500)")
    parser.add_argument("--n_runs",          type=int, default=3,
                        help="Number of repeat runs per experiment (default: 3)")
    parser.add_argument("--chunk_size",      type=int, default=100,
                        help="Images per batch request (default: 100)")
    parser.add_argument("--thinking_budget", type=int, default=4096,
                        help="Max thinking tokens per request (default: 2048)")
    args = parser.parse_args()

    # ── Experiment loop ───────────────────────────────────────────────────────

    for exp in args.experiments:

        if exp == 'RQ1':
            experiment_plan = {
                'sub_experiment_1': {'img': 'images', 'qa': 'qa.json'}
            }
        else:
            experiment_plan = {
                'sub_experiment_1': {'img': 'images_numbers', 'qa': 'qa_numbers.json'},
                'sub_experiment_2': {'img': 'images_letters', 'qa': 'qa_letters.json'},
                'sub_experiment_3': {'img': 'images_dots',    'qa': 'qa_dots.json'},
            }

        exp_dir = os.path.join(args.data_dir, exp)

        for sub_experiment, data in experiment_plan.items():

            selected_image   = data['img']
            selected_qa      = data['qa']
            qa_file_path     = os.path.join(exp_dir, selected_qa)
            image_files_path = os.path.join(exp_dir, selected_image)

            with open(qa_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            png_images = [entry['filename'] for entry in json_data if 'filename' in entry]

            random.seed(2025)
            N = args.n_images

            if N > len(png_images):
                print(f"Selected N={N} > available {len(png_images)}. Exiting.")
                sys.exit(0)
            elif N == len(png_images):
                print(f"Using whole dataset ({N} images).")
                mo_file_name_appendix = 'all_images'
            else:
                print(f"Random pick: {N} images.")
                png_images = random.sample(png_images, N)
                mo_file_name_appendix = f'random_pick_{N}_images'

            for j in range(args.n_runs):
                start_time = time.time()
                dataset_results = []

                for i in range(0, len(png_images), args.chunk_size):
                    chunk = png_images[i: i + args.chunk_size]
                    print(f"{exp} | run {j} | chunk {i // args.chunk_size + 1}: "
                          f"images {i}–{i + len(chunk) - 1}")

                    batch_messages = []
                    batch_metadata = []

                    for image in chunk:
                        question_data = get_qa(image, qa_file_path)

                        other_images = [img for img in chunk if img != image]
                        additional_question = get_qa(random.choice(other_images), qa_file_path) \
                            if other_images else None

                        base64_image = get_clean_image(
                            os.path.join(image_files_path, image))

                        prompt = (
                            "The image is a 2D axial slice of an abdominal CT scan with soft tissue windowing. "
                            "Answer strictly with '1' for Yes or '0' for No. No explanations, no additional text. "
                            "Your output must contain exactly one character: '1' or '0'. "
                            "Ignore anatomical correctness; focus solely on what the image shows.\n"
                            "Example:\n"
                            f"Q: {additional_question[0]['question']} A: {additional_question[0]['answer']}\n"
                            "Now answer the real question:\n\n"
                            f"Q: {question_data[0]['question']}"
                        )

                        batch_messages.append([
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url",
                                     "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                ]
                            }
                        ])
                        batch_metadata.append({
                            "file_name":       image,
                            "question":        question_data[0]['question'],
                            "expected_answer": question_data[0]['answer'],
                            "entire_prompt":   prompt,
                        })

                    # ── Single batch request for the whole chunk ───────────────
                    responses = send_batch(
                        batch_messages,
                        server_url=args.server_url,
                        model_name=args.model_name,
                        thinking_budget=args.thinking_budget,
                    )

                    # Debug: show first 2 results
                    for resp in responses[:2]:
                        text, reasoning = parse_output(resp)
                        print("RAW TEXT:    ", repr(text))
                        print("REASONING:   ", repr(reasoning[:200]) if reasoning else None)
                        print("FINISH REASON:", resp["choices"][0]["finish_reason"])

                    for metadata, response in zip(batch_metadata, responses):
                        text, reasoning = parse_output(response)
                        tokens_used = response.get("usage", {}).get("completion_tokens", 0)

                        dataset_results.append({
                            "file_name": metadata["file_name"],
                            "results_call": [{
                                "question":        metadata["question"],
                                "model_answer":    text,
                                "thinking":        reasoning,
                                "tokens_used":     tokens_used,
                                "expected_answer": metadata["expected_answer"],
                                "entire_prompt":   metadata["entire_prompt"],
                            }]
                        })

                results_file_name = (
                    f"{exp}_{selected_qa.replace('.json', '')}"
                    f"_{mo_file_name_appendix}_add_run_{j}.json"
                )
                save_name = os.path.join(args.results_dir, results_file_name)
                os.makedirs(os.path.dirname(save_name), exist_ok=True)

                with open(save_name, 'w') as f:
                    json.dump(dataset_results, f, indent=4)

                elapsed = time.time() - start_time
                print(f"Runtime for {selected_qa.replace('.json', '')} "
                      f"with {selected_image}: {elapsed:.2f}s")
