import os
import sys
import json
import time
import random
import base64
import argparse
import asyncio
import aiohttp

from io import BytesIO
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
#  Image helpers (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def encode_image_from_bytes(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_grayscale(image):
    return image.mode in ["L", "I;16"]


def normalize_16bit_to_8bit(image):
    normalized_image = image.point(lambda x: (x / 256))
    return normalized_image.convert("L")


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
    target_filename = img_file_name
    result = next((entry['question_answer']
                   for entry in data if entry['filename'] == target_filename), None)
    return [{'question': entry['question'], 'answer': entry['answer']} for entry in result]


# ──────────────────────────────────────────────────────────────────────────────
#  Server inference helpers
# ──────────────────────────────────────────────────────────────────────────────

SERVER_URL = "http://localhost:8001/v1/chat/completions"
MODEL_NAME = "models/Qwen3.5-9B"  # must match --served-model-name on the server


async def send_single(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                      messages: list, thinking_budget: int = 2048) -> dict:
    """Send one chat request to the vLLM server and return the parsed response."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 4096,
        # Qwen3 thinking budget via extra_body
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
    }

    async with semaphore:
        async with session.post(SERVER_URL, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def run_batch(batch_messages: list[list], max_concurrent: int = 32) -> list[dict]:
    """Send all messages in parallel (up to max_concurrent at a time)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=300)  # 5 min per request

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [send_single(session, semaphore, msgs) for msgs in batch_messages]
        return await asyncio.gather(*tasks)


def parse_output(response: dict) -> tuple[str, str | None]:
    """Extract (text, reasoning_content) from a /v1/chat/completions response."""
    choice = response["choices"][0]
    message = choice["message"]
    text = message.get("content", "") or ""
    # The vLLM online server puts thinking in reasoning_content when
    # --enable-reasoning / reasoning-parser is active
    reasoning = message.get("reasoning_content", None)
    return text, reasoning


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test run Script (server mode)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--max_concurrent", type=int, default=32,
                        help="Max parallel requests to the vLLM server (default: 32)")
    args = parser.parse_args()

    dataset_dir = args.data_dir

    RESULTS_ROOT = 'control/results/qwen3.5_thinking'
    CHUNK_SIZE = 500

    experiments = ['RQ1']

    for exp in experiments:

        if exp == 'RQ1':
            experiment_plan = {
                'sub_experiment_1': {'img': 'images', 'qa': 'qa.json'}
            }
        else:
            experiment_plan = {
                'sub_experiment_1': {'img': 'images_numbers',  'qa': 'qa_numbers.json'},
                'sub_experiment_2': {'img': 'images_letters',  'qa': 'qa_letters.json'},
                'sub_experiment_3': {'img': 'images_dots',     'qa': 'qa_dots.json'},
            }

        exp_dir = os.path.join(dataset_dir, exp)

        for sub_experiment, data in experiment_plan.items():

            selected_image = data['img']
            selected_qa    = data['qa']

            qa_file_path      = os.path.join(exp_dir, selected_qa)
            image_files_path  = os.path.join(exp_dir, selected_image)

            with open(qa_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            png_images = [entry['filename'] for entry in json_data if 'filename' in entry]

            random.seed(2025)
            N = 500

            if N > len(png_images):
                print(f'Selected N={N} > available {len(png_images)}. Exiting.')
                sys.exit(0)
            elif N == len(png_images):
                print(f'Using whole dataset ({N} images).')
                mo_file_name_appendix = 'all_images'
            else:
                print(f'Random pick: {N} images.')
                png_images = random.sample(png_images, N)
                mo_file_name_appendix = f'random_pick_{N}_images'

            for j in range(3):
                start_time = time.time()
                dataset_results = []

                for i in range(0, len(png_images), CHUNK_SIZE):
                    chunk = png_images[i: i + CHUNK_SIZE]
                    print(f"{exp} Run {j} | chunk {i // CHUNK_SIZE + 1}: "
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
                            "Your output must contain exactly one character: '1' or '0'."
                            "Ignore anatomical correctness; focus solely on what the image shows.\n"
                            "Example:\n"
                            f"Q: {additional_question[0]['question']} A: {additional_question[0]['answer']}\n"
                            "Now answer the real question:\n\n"
                            f"Q: {question_data[0]['question']}"
                        )

                        msg = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url",
                                     "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                ]
                            }
                        ]

                        batch_messages.append(msg)
                        batch_metadata.append({
                            "file_name": image,
                            "question": question_data[0]['question'],
                            "expected_answer": question_data[0]['answer'],
                            "entire_prompt": prompt,
                        })

                    # ── Fire the whole chunk concurrently ──────────────────────
                    responses = asyncio.run(
                        run_batch(batch_messages, max_concurrent=args.max_concurrent)
                    )

                    # Debug: show first 2 raw responses
                    for resp in responses[:2]:
                        text, reasoning = parse_output(resp)
                        print("RAW TEXT:", repr(text))
                        print("REASONING:", repr(reasoning[:200]) if reasoning else None)
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
                save_name = os.path.join(RESULTS_ROOT, results_file_name)
                os.makedirs(os.path.dirname(save_name), exist_ok=True)

                with open(save_name, 'w') as f:
                    json.dump(dataset_results, f, indent=4)

                elapsed = time.time() - start_time
                print(f"Runtime for {selected_qa.replace('.json', '')} "
                      f"with {selected_image}: {elapsed:.2f}s")
