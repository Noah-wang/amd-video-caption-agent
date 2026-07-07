import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from openai import OpenAI


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/qwen2p5-vl-32b-instruct"
FIREWORKS_TEXT_MODEL = "accounts/fireworks/models/qwen2p5-32b-instruct"
THEBESTAI_BASE_URL = "https://thebestai.net/v1"
THEBESTAI_VISION_MODEL = "glm-4v-plus"
THEBESTAI_TEXT_MODEL = "deepseek-v4-flash-none"


def resolve_input_path() -> Path:
    docker_input = Path("/input/tasks.json")
    local_input = Path("input/tasks.json")
    return docker_input if docker_input.exists() else local_input


def resolve_output_path() -> Path:
    docker_output_dir = Path("/output")
    return (
        docker_output_dir / "results.json"
        if docker_output_dir.exists()
        else Path("output/results.json")
    )


def download_video(video_url: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="video_caption_"))
    video_path = temp_dir / "clip.mp4"
    urllib.request.urlretrieve(video_url, video_path)
    print(f"Downloaded video to: {video_path}")
    return video_path


def get_video_duration(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_frames(video_path: Path, frame_count: int = 4) -> list[Path]:
    frames_dir = Path("samples/frames")

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video_path)
    print(f"Video duration: {duration:.2f} seconds")

    frame_paths = []
    for index in range(frame_count):
        timestamp = duration * (index + 1) / (frame_count + 1)
        output_path = frames_dir / f"frame_{index + 1:03d}.jpg"

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.2f}",
            "-i",
            str(video_path),
            "-vf",
            "scale=768:-1",
            "-frames:v",
            "1",
            str(output_path),
        ]

        subprocess.run(command, check=True, capture_output=True, text=True)
        frame_paths.append(output_path)

    print("Extracted frames:")
    for frame_path in frame_paths:
        print(f"  - {frame_path}")

    return frame_paths


def get_provider() -> str:
    requested_provider = os.getenv("AI_PROVIDER", "").strip().lower()
    if requested_provider:
        return requested_provider
    if os.getenv("FIREWORKS_API_KEY"):
        return "fireworks"
    if os.getenv("THEBESTAI_API_KEY"):
        return "thebestai"
    return "fireworks"


def create_ai_client() -> OpenAI:
    provider = get_provider()
    if provider == "fireworks":
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "FIREWORKS_API_KEY is not set. Export it before running the program."
            )
        base_url = os.getenv("FIREWORKS_BASE_URL", FIREWORKS_BASE_URL)
    elif provider == "thebestai":
        api_key = os.getenv("THEBESTAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "THEBESTAI_API_KEY is not set. Export it before running the program."
            )
        base_url = os.getenv("THEBESTAI_BASE_URL", THEBESTAI_BASE_URL)
    else:
        raise RuntimeError(f"Unsupported AI_PROVIDER: {provider}")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60,
    )


def get_model(kind: str) -> str:
    provider = get_provider()
    if provider == "fireworks":
        if kind == "vision":
            return os.getenv("FIREWORKS_VISION_MODEL", FIREWORKS_VISION_MODEL)
        return os.getenv("FIREWORKS_TEXT_MODEL", FIREWORKS_TEXT_MODEL)
    if provider == "thebestai":
        if kind == "vision":
            return os.getenv("THEBESTAI_VISION_MODEL", THEBESTAI_VISION_MODEL)
        return os.getenv("THEBESTAI_TEXT_MODEL", THEBESTAI_TEXT_MODEL)
    raise RuntimeError(f"Unsupported AI_PROVIDER: {provider}")


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def describe_video_frames(frame_paths: list[Path]) -> str:
    client = create_ai_client()
    model = get_model("vision")

    content = [
        {
            "type": "text",
            "text": (
                "You are looking at 4 frames sampled from a short video. "
                "Write a concise, factual 1-2 sentence description of what is happening. "
                "Focus on visible subjects, actions, and setting. "
                "Do not use humor. Do not speculate beyond what is visible."
            ),
        }
    ]

    for frame_path in frame_paths:
        image_base64 = encode_image_base64(frame_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=160,
    )

    summary = get_message_text(response.choices[0].message)
    print(f"Visual summary: {summary}")
    return summary


def get_message_text(message) -> str:
    content = message.content or ""
    reasoning_content = getattr(message, "reasoning_content", None) or ""
    return content.strip() or reasoning_content.strip()


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def first_words(text: str, limit: int = 16) -> str:
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    return " ".join(words[:limit]).rstrip(".,;:") + "."


def fallback_caption(style: str, visual_summary: str) -> str:
    short_summary = first_words(visual_summary)
    if style == "sarcastic":
        return f"Oh look, another scene of {short_summary.lower()}"
    if style == "humorous_tech":
        return f"System log captured: {short_summary}"
    if style == "humorous_non_tech":
        return f"Somehow this ordinary scene became entertaining: {short_summary}"
    return short_summary


def rewrite_captions(styles: list[str], visual_summary: str) -> dict[str, str]:
    client = create_ai_client()
    model = get_model("text")

    style_instructions = {
        "formal": "Use a professional, objective, factual tone.",
        "sarcastic": "Use a dry, ironic, lightly mocking tone.",
        "humorous_tech": (
            "Use humor with clear technology or programming references, such as "
            "code, bugs, servers, latency, algorithms, UI, APIs, or system logs."
        ),
        "humorous_non_tech": "Use everyday humor with no technical jargon.",
    }

    requested_styles = {
        style: style_instructions.get(
            style,
            "Use a concise, factual tone while preserving the video facts.",
        )
        for style in styles
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite factual video summaries into short captions. "
                    "Preserve the visible facts. Do not invent people, objects, "
                    "actions, locations, or dialogue. Return only a valid JSON object. "
                    "Use plain ASCII punctuation. Do not include markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Requested styles as JSON: {json.dumps(requested_styles)}\n"
                    f"Video facts: {visual_summary}\n"
                    "Return a JSON object whose keys exactly match the requested styles. "
                    "Each value must be one English caption in 8 to 20 words."
                ),
            },
        ],
        temperature=0.6,
        max_tokens=240,
    )

    text = get_message_text(response.choices[0].message)
    try:
        generated = parse_json_object(text)
    except json.JSONDecodeError:
        generated = {}

    captions = {}
    for style in styles:
        caption = generated.get(style)
        if isinstance(caption, str) and caption.strip():
            captions[style] = caption.strip()
        else:
            captions[style] = fallback_caption(style, visual_summary)

    return captions


def main() -> None:
    input_path = resolve_input_path()
    output_path = resolve_output_path()

    tasks = json.loads(input_path.read_text())

    results = []
    for task in tasks:
        video_path = download_video(task["video_url"])
        frame_paths = extract_frames(video_path)
        visual_summary = describe_video_frames(frame_paths)

        captions = rewrite_captions(task["styles"], visual_summary)

        results.append(
            {
                "task_id": task["task_id"],
                "captions": captions,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
