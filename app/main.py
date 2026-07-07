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
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/minimax-m3"
FIREWORKS_TEXT_MODEL = "accounts/fireworks/models/gpt-oss-120b"
THEBESTAI_BASE_URL = "https://thebestai.net/v1"
THEBESTAI_VISION_MODEL = "glm-4v-plus"
THEBESTAI_TEXT_MODEL = "deepseek-v4-flash-none"
TECH_TERMS = [
    "api",
    "bug",
    "server",
    "latency",
    "algorithm",
    "ui",
    "code",
    "system",
    "log",
    "render",
    "input",
    "cache",
    "runtime",
    "deploy",
    "update",
    "software",
    "firmware",
    "wifi",
    "gps",
]


def load_local_env(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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
                "These are sequential frames sampled from one video, not a collage. "
                "Describe the video scene, not the frame layout. Output only one "
                "final factual sentence. Focus on visible subjects, actions, and "
                "setting. No analysis, bullets, markdown, or instructions."
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
        temperature=0.1,
        max_tokens=2000,
    )

    summary = clean_visual_summary(get_message_text(response.choices[0].message))
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


def clean_visual_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text

    try:
        parsed = parse_json_object(text)
        summary = parsed.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    except json.JSONDecodeError:
        pass

    quoted_sentences = re.findall(r'"([^"]{30,240})"', text)
    if quoted_sentences:
        return quoted_sentences[-1].strip()

    blocked_phrases = [
        "analyze the request",
        "professional image captioning",
        "grounding:",
        "role:",
        "length:",
        "focus:",
        "no analysis",
    ]
    useful_starts = [
        "The image shows",
        "The images show",
        "The image depicts",
        "The images depict",
        "A ",
        "An ",
    ]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    candidates = []
    for sentence in sentences:
        stripped = sentence.strip("-• ").strip()
        lower = stripped.lower()
        if 30 <= len(stripped) <= 260 and any(
            stripped.startswith(start) for start in useful_starts
        ) and not any(phrase in lower for phrase in blocked_phrases):
            candidates.append(stripped)

    if candidates:
        return candidates[-1]

    if any(phrase in text.lower() for phrase in blocked_phrases):
        return "A short video scene with visible subjects, actions, and setting."

    return first_words(text, 32)


def first_words(text: str, limit: int = 16) -> str:
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    return " ".join(words[:limit]).rstrip(".,;:") + "."


def normalize_caption(text: str) -> str:
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "‑": "-",
        "…": "...",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def fallback_caption(style: str, visual_summary: str) -> str:
    short_summary = first_words(visual_summary, 18)
    if style == "sarcastic":
        return f"Oh look, another scene of {short_summary.lower()}"
    if style == "humorous_tech":
        return f"System log captured this scene with zero extra processing required."
    if style == "humorous_non_tech":
        return f"Somehow this ordinary scene still found a way to be entertaining."
    return short_summary


def has_tech_term(text: str) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lower) for term in TECH_TERMS)


def choose_caption(style: str, caption: object, visual_summary: str) -> str:
    if isinstance(caption, str) and caption.strip():
        normalized = normalize_caption(caption)
        is_reasonable_length = 4 <= word_count(normalized) <= 30
        has_matching_tech_tone = (
            (style == "humorous_tech" and has_tech_term(normalized))
            or (style == "humorous_non_tech" and not has_tech_term(normalized))
            or style not in {"humorous_tech", "humorous_non_tech"}
        )
        if is_reasonable_length and has_matching_tech_tone:
            return normalized
    return normalize_caption(fallback_caption(style, visual_summary))


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
                    "You are a JSON-only caption generator. Output the final "
                    "answer only. Do not explain. Preserve visible facts and "
                    "use plain ASCII punctuation."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create captions for this video summary. Output ONLY a "
                    "minified JSON object, no markdown. Keys must exactly match "
                    f"this JSON object: {json.dumps(requested_styles)}\n"
                    "Each value must be one short English caption in 8 to 16 words. "
                    "Use one sentence per caption. Avoid long clauses. "
                    "Do not invent people, objects, actions, locations, or dialogue.\n"
                    f"Video summary: {visual_summary}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    text = get_message_text(response.choices[0].message)
    try:
        generated = parse_json_object(text)
    except json.JSONDecodeError:
        generated = {}

    captions = {}
    for style in styles:
        captions[style] = choose_caption(style, generated.get(style), visual_summary)

    return captions


def main() -> None:
    load_local_env()

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
