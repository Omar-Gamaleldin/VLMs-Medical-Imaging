import os
import json
import random
import base64
import argparse
import litellm
from gepa.adapters.default_adapter.default_adapter import (
    DefaultAdapter,
    DefaultDataInst,
    EvaluationResult,
)
import gepa
from io import BytesIO
from PIL import Image


def encode_image_from_bytes(image):
    """
    Encodes an image object into a base64-encoded PNG string.

    Args:
        image (PIL.Image.Image): The image to encode.

    Returns:
        str: Base64-encoded string representation of the image.

    Example:
        ```python
        from PIL import Image
        import base64

        img = Image.open("example.png")
        encoded_string = encode_image_from_bytes(img)
        print(encoded_string)
        ```
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_grayscale(image):
    """
    Checks if a given image is in grayscale mode.

    Args:
        image (PIL.Image.Image): The image to check.

    Returns:
        bool: True if the image is grayscale, False otherwise.

    The function checks whether the image mode is either "L" (8-bit grayscale) 
    or "I;16" (16-bit grayscale).

    Example:
        ```python
        from PIL import Image

        img = Image.open("example.png")
        if is_grayscale(img):
            print("The image is grayscale.")
        else:
            print("The image is in color.")
        ```
    """
    return image.mode in ["L", "I;16"]


def normalize_16bit_to_8bit(image):
    """
    Normalizes a 16-bit grayscale image to an 8-bit grayscale image.

    Args:
        image (PIL.Image.Image): A 16-bit grayscale image (mode "I;16").

    Returns:
        PIL.Image.Image: An 8-bit grayscale image (mode "L").

    The function scales pixel values from the 16-bit range (0-65535) to the 
    8-bit range (0-255) by dividing each pixel by 256 and then converting 
    the image to mode "L".

    Example:
        ```python
        from PIL import Image

        img_16bit = Image.open("example_16bit.png")
        img_8bit = normalize_16bit_to_8bit(img_16bit)
        img_8bit.save("example_8bit.png")
        ```
    """
    normalized_image = image.point(
        lambda x: (x / 256))
    return normalized_image.convert("L")


def ensure_rgb(image):
    """
    Ensures that the given image is in RGB mode.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The image converted to RGB mode.

    This function handles different image modes as follows:
    - If the image is in "I;16" mode (16-bit grayscale), it is first normalized 
      to 8-bit grayscale and then converted to RGB.
    - If the image is grayscale ("L") or has an alpha channel ("RGBA"), 
      it is directly converted to RGB.
    - If the image is already in "RGB" mode, a copy is returned.
    - If the image mode is unsupported, a `ValueError` is raised.

    Example:
        ```python
        from PIL import Image

        img = Image.open("example.png")
        rgb_img = ensure_rgb(img)
        rgb_img.show()
        ```

    Raises:
        ValueError: If the image mode is not supported.
    """
    if image.mode == "I;16":
        return normalize_16bit_to_8bit(image).convert("RGB")
    elif is_grayscale(image) or image.mode == "RGBA":
        return image.convert("RGB")
    elif image.mode == "RGB":
        return image.copy()
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")


def get_clean_image(image_path):
    """
    Loads an image, ensures it is in RGB mode, and encodes it as a base64 string.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: The base64-encoded representation of the image.

    This function performs the following steps:
    1. Opens the image from the given path.
    2. Converts it to RGB mode if necessary using `ensure_rgb()`.
    3. Encodes the processed image into a base64 string using `encode_image_from_bytes()`.

    Example:
        ```python
        encoded_image = get_clean_image("example.png")
        print(encoded_image)
        ```
    """
    with Image.open(image_path) as img:
        rgb_image = ensure_rgb(img)
    base64_image = encode_image_from_bytes(rgb_image)

    return base64_image

def get_qa(img_file_name, json_dir):
    """
    Retrieves the question-answer pairs for a given image file from a JSON dataset.

    Args:
        img_file_name (str): The filename of the image for which QA pairs are required.
        json_dir (str): The path to the JSON file containing question-answer data.

    Returns:
        list[dict]: A list of dictionaries, each containing a 'question' and an 'answer'.

    The function performs the following steps:
    1. Loads the JSON file from the provided directory.
    2. Finds the entry that matches the given image filename.
    3. Extracts and returns the associated question-answer pairs.

    Example:
        ```python
        qa_pairs = get_qa("image_001.jpg", "questions.json")
        for qa in qa_pairs:
            print(f"Q: {qa['question']}\nA: {qa['answer']}")
        ```
    """
    with open(json_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)

    target_filename = img_file_name
    result = next((entry['question_answer']
                   for entry in data if entry['filename'] == target_filename), None)

    questions_answers = [{'question': entry['question'],
                          'answer': entry['answer']} for entry in result]
    return questions_answers

# ──────────────────────────────────────────────────────────────────────────────
#  Setting up GEPA Classes and Funstions
# ──────────────────────────────────────────────────────────────────────────────

VLM_MODEL= "openai/qwen3.5-9b"
REFLECTIVE_MODEL= "openai/qwen3.5-9b"

class BinaryAnswerEvaluator:
    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        expected = str(data["answer"])  # cast int 0/1 to "0"/"1"
        is_correct = expected in response
        score = 1.0 if is_correct else 0.0
        if is_correct:
            feedback = f"Correct. Response contained the expected answer '{expected}'."
        else:
            feedback = (
                f"Incorrect. Expected '{expected}' but got: '{response}'. "
                f"Respond with only '1' or '0'."
            )
        return EvaluationResult(score=score, feedback=feedback)


def chat_callable(messages) -> str:
    resp = litellm.completion(
        model=VLM_MODEL,
        messages=messages,
        api_base="http://localhost:8001/v1",
        api_key="dummy",
        stream=False
    )
    content = resp.choices[0].message.content
    if content is None:
        content = getattr(resp.choices[0].message, "reasoning_content", "") or ""
    return content.strip()


def reflection_lm_callable(prompt) -> str:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    resp = litellm.completion(
        model=REFLECTIVE_MODEL,
        messages=messages,
        api_base="http://localhost:8001/v1",
        api_key="dummy",
    )
    content = resp.choices[0].message.content
    if content is None:                        # addition
        content = getattr(resp.choices[0].message, "reasoning_content", "") or ""
    return content.strip()


if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────────────────
    #  Parsing Directories
    # ──────────────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Test run Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, "RQ1")
    images_dir = os.path.join(data_dir, "images")
    qa_path = os.path.join(data_dir, "qa.json")

    # ──────────────────────────────────────────────────────────────────────────────
    #  Setting up subset of MIRP Benchmark
    # ──────────────────────────────────────────────────────────────────────────────

    with open(qa_path, "r", encoding="utf8") as file:
        data = json.load(file)

    random.seed(2026)

    entire_dataset = [
            {
                "input": "/no-think" + entry["question_answer"][0]["question"], 
                "answer": str(entry["question_answer"][0]["answer"]),
                "additional_context": {"image": entry["filename"]}
            } 
            for entry in data
        ]

    dataset = random.choices(entire_dataset, k=200)

    images = [entry["additional_context"]["image"] for entry in dataset]
    with open("excluded_images.txt", "w", encoding="utf-8") as f:
        f.writelines([img + "\n" for img in images])

    for img in images:
        img_dir = os.path.join(images_dir, img)
        rgb_image = get_clean_image(img_dir)

        for i, entry  in enumerate(dataset):
            if entry["additional_context"]["image"] == img:
                dataset[i]["additional_context"]["image"] = rgb_image
                break

    # ──────────────────────────────────────────────────────────────────────────────
    # Initializing GEPA Optimisation
    # ──────────────────────────────────────────────────────────────────────────────
    seed_prompt = {
        "system_prompt": "You are presented with CT scans. Answer the questions concisly with either a '1' or '0'. Don't give any explanations."
    }

    adapter = DefaultAdapter(
        model=chat_callable,
        evaluator=BinaryAnswerEvaluator(),
    )

    # Run optimization
    result = gepa.api.optimize(
        seed_candidate=seed_prompt,
        trainset=dataset[:160],
        valset=dataset[160:],
        task_lm=chat_callable,                  # Model to optimize
        reflection_lm=reflection_lm_callable,   # Model for reflection
        evaluator=BinaryAnswerEvaluator()
        max_metric_calls=2190,                  # up to 50 iterations
        stop_callbacks=TimeoutStopCondition(seconds=3 * 3600),
    )

    # Get the optimized prompt and best score
    print("Best prompt:", result.best_candidate['system_prompt'])
    print("Best score:", result.val_aggregate_scores[result.best_idx])


