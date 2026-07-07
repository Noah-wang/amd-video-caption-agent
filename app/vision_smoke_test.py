import base64
import os
import time
from pathlib import Path

from openai import OpenAI


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def get_message_text(message) -> str:
    content = message.content or ""
    reasoning_content = getattr(message, "reasoning_content", None) or ""
    return content.strip() or reasoning_content.strip()


def main() -> None:
    api_key = os.getenv("THEBESTAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "THEBESTAI_API_KEY is not set. Export it before running this script."
        )

    model = os.getenv("THEBESTAI_VISION_MODEL", "glm-4.6v")
    image_path = Path("samples/frames/frame_001.jpg")

    if not image_path.exists():
        raise RuntimeError(
            "samples/frames/frame_001.jpg does not exist. Run python app/main.py first."
        )

    print(f"Testing model: {model}")
    print(f"Image path: {image_path}")
    print(f"Image size: {image_path.stat().st_size} bytes")

    client = OpenAI(
        api_key=api_key,
        base_url="https://thebestai.net/v1",
        timeout=60,
    )

    image_base64 = encode_image_base64(image_path)

    started_at = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a visual captioning model. "
                    "Do not show reasoning. "
                    "Only output the final answer."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in one short factual English sentence. "
                            "Return only the sentence."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ],
        temperature=0.1,
        max_tokens=200,
    )
    elapsed = time.time() - started_at

    message = response.choices[0].message
    text = get_message_text(message)

    print(f"Elapsed: {elapsed:.2f}s")
    print("Raw response:")
    print(response.model_dump_json(indent=2))
    print("Parsed text:")
    print(text)


if __name__ == "__main__":
    main()
