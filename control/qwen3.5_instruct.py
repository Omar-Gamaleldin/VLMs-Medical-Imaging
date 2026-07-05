import os
import sys
import json
import time
import random
import base64
import argparse
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM
from vllm.sampling_params import SamplingParams
from io import BytesIO
from PIL import Image


def encode_image_from_bytes(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_grayscale(image):
    return image.mode in ["L", "I;16"]


def normalize_16bit_to_8bit(image):
    normalized_image = image.point(
        lambda x: (x / 256))
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
    base64_image = encode_image_from_bytes(rgb_image)

    return base64_image


def get_qa(img_file_name, json_dir):
    with open(json_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)

    target_filename = img_file_name
    result = next((entry['question_answer']
                   for entry in data if entry['filename'] == target_filename), None)

    questions_answers = [{'question': entry['question'],
                          'answer': entry['answer']} for entry in result]
    return questions_answers



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test run Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--excluded_qs", type=str, default="", help="File containing excluded questions")
    parser.add_argument("--prompt_file", type=str, default="", help="File containing gepa optimized prompt")
    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────────────────
    #  Model
    # ──────────────────────────────────────────────────────────────────────────────

    model_dir = "models/Qwen3.5-9B"

    sampling_params = SamplingParams(max_tokens=2)
    llm = LLM(
        model=model_dir,
        gpu_memory_utilization=0.95,  # Maximale GPU-Nutzung
        trust_remote_code=True
    )
    print("Finished loading the model")
    # ──────────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────────
    #  Paths and Experiment Selection
    # ──────────────────────────────────────────────────────────────────────────────
    
    dataset_dir = os.path.join(args.data_dir)
    excluded_dir = os.path.join("gepa/", args.excluded_qs)
    prompt_dir = os.path.join("gepa/", args.prompt_file)

    RESULTS_ROOT = 'control/results/qwen3.5/gepa'  # path for results directory
    CHUNK_SIZE = 500

    experiments = ['RQ1']  # select the experiments here: 'RQ1', 'RQ2', 'RQ3', 'AS'
    # ──────────────────────────────────────────────────────────────────────────────

    for exp in experiments:

        if exp == 'RQ1':
            experiment_plan = {
                'sub_experiment_1': {'img': 'images',
                                     'qa': 'qa.json'}
            }

        else:
            experiment_plan = {
                'sub_experiment_1': {'img': 'images_numbers',
                                     'qa': 'qa_numbers.json'},
                'sub_experiment_2': {'img': 'images_letters',
                                     'qa': 'qa_letters.json'},
                'sub_experiment_3': {'img': 'images_dots',
                                     'qa': 'qa_dots.json'}
            }

        exp_dir = os.path.join(dataset_dir, exp)

        excluded = []
        if args.excluded_qs != "":
            with open(excluded_dir , 'r', encoding='utf-8') as file:
                all_files = file.readlines()
            print(f"Numberof images excluded: {len(all_files)}")
            excluded = [file.strip() for file in all_files]
            print(f"Numberof images excluded: {len(excluded)}")

        gepa_prompt = ""
        if args.prompt_file != "":
            with open(prompt_dir , 'r', encoding='utf-8') as file:
                gepa_prompt = file.read()

        for sub_experiment, data in experiment_plan.items():

            selected_image = data['img']
            selected_qa = data['qa']

            qa_file_path = os.path.join(exp_dir, selected_qa)

            image_files_path = os.path.join(exp_dir, selected_image)

            with open(qa_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            png_images = [entry['filename']
                          for entry in data if 'filename' in entry and entry['filename'] not in excluded]

            random.seed(2025)

            N = len(png_images)  # number or len(png_images)

            if N > len(png_images):
                print(f'The selected amount of images {N} is bigger than the available images {len(png_images)}.')
                sys.exit(0)
            elif N == len(png_images):
                print(f'The selected amount of images {N} is equal to the available images {len(png_images)}. Not picking random, using whole dataset instead.')
                mo_file_name_appendix = 'all_images'
                png_images = png_images
            else:
                print(f'Using random pick with {N} images.')
                png_images = random.sample(png_images, N)
                mo_file_name_appendix = f'random_pick_{N}_images'

            for j in range(3):  # how many runs ?
                start_time = time.time()

                dataset_results = []

                for i in range(0, len(png_images), CHUNK_SIZE):
                    batch_messages = []
                    batch_metadata = []

                    chunk = png_images[i : i + CHUNK_SIZE]
                    print(f"{exp} Run {j} Processing chunk {i//CHUNK_SIZE + 1}: images {i} to {i + len(chunk)}")

                    for image in chunk:
                        question_data = get_qa(image, qa_file_path)

                        other_images = [
                            img for img in chunk if img != image]
                        if other_images:
                            random_other_image = random.choice(other_images)
                            additional_question = get_qa(
                                random_other_image, qa_file_path)
                        else:
                            additional_question = None

                        original_image_path = os.path.join(
                            image_files_path, image)

                        base64_image = get_clean_image(original_image_path)

                        prompt = (
                            "The image is a 2D axial slice of an abdominal CT scan with soft tissue windowing. "
                            "Answer strictly with '1' for Yes or '0' for No. No explanations, no additional text. "
                            "Your output must contain exactly one character: '1' or '0'."
                            "Ignore anatomical correctness; focus solely on what the image shows.\n"
                            "Example:\n"
                            # dynamic part of the prompt
                            f"Q: {additional_question[0]['question']} A: {additional_question[0]['answer']}\n"
                            "Now answer the real question:\n\n"
                            f"Q: {question_data[0]['question']}"
                        ) if args.prompt_file == "" else (
                            f"{gepa_prompt}"
                            "Now answer the real question:\n\n"
                            f"Q: {question_data[0]['question']}"
                            )
                        msg = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                ]
                            },
                        ]

                        batch_messages.append(msg)
                        batch_metadata.append({
                            "file_name": image,
                            "question": question_data[0]['question'],
                            "expected_answer": question_data[0]['answer'],
                            "entire_prompt": prompt
                        })

                    outputs = llm.chat(batch_messages, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False})
                    print(outputs[0].outputs)

                    for metadata, model_output in zip(batch_metadata, outputs):
                            dataset_results.append({
                                "file_name": metadata["file_name"],
                                "results_call" : [{
                                    "question": metadata["question"],
                                    "model_answer": model_output.outputs[0].text,
                                    "expected_answer": metadata["expected_answer"],
                                    "entire_prompt": metadata["entire_prompt"]
                                    }]
                            })

                results_file_name = f"{exp}_{selected_qa.replace('.json', '')}_{mo_file_name_appendix}_add_run_{j}.json"

                save_name = os.path.join(
                    RESULTS_ROOT, results_file_name)

                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_name), exist_ok=True)

                with open(save_name, 'w') as json_file:
                    json.dump(dataset_results, json_file, indent=4)
                end_time = time.time()

                elapsed_time = end_time - start_time
                print(f"Runtime for {selected_qa.replace('.json', '')} with {selected_image} : {elapsed_time:.2f} seconds")
