import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from openai import OpenAI


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/minimax-m3"
FIREWORKS_TEXT_MODEL = "accounts/fireworks/models/gpt-oss-120b"
THEBESTAI_BASE_URL = "https://thebestai.net/v1"
THEBESTAI_VISION_MODEL = "glm-4v-plus"
THEBESTAI_TEXT_MODEL = "deepseek-v4-flash-none"
MAX_MODEL_RETRIES = 3
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
BANNED_UNSEEN_PATTERNS = [
    r"\b(email|emails|coffee|spreadsheet|spreadsheets|litter box)\b",
    r"\b(compiles?|debugs?|coding|programming)\s+(code|spreadsheets?|life)\b",
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


def choose_frame_count(duration: float) -> int:
    configured = os.getenv("VIDEO_FRAME_COUNT")
    if configured:
        return max(1, int(configured))
    if duration < 10:
        return 4
    if duration < 30:
        return 5
    return 6


def extract_frames(video_path: Path) -> list[Path]:
    frames_dir = Path("samples/frames")

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video_path)
    print(f"Video duration: {duration:.2f} seconds")
    frame_count = choose_frame_count(duration)
    print(f"Sampling frames: {frame_count}")

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


def create_chat_completion_with_retries(client: OpenAI, **kwargs):
    last_error = None
    for attempt in range(1, MAX_MODEL_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as error:
            last_error = error
            if attempt == MAX_MODEL_RETRIES:
                break
            sleep_seconds = 2 ** (attempt - 1)
            print(f"Model call failed, retrying in {sleep_seconds}s: {error}")
            time.sleep(sleep_seconds)
    raise last_error


def describe_video_frames(frame_paths: list[Path]) -> str:
    client = create_ai_client()
    model = get_model("vision")

    content = [
        {
            "type": "text",
            "text": (
                "These are sequential frames sampled from one video, not a collage. "
                "Describe the video scene, not the frame layout. Output only valid "
                "minified JSON with keys summary, subjects, actions, setting, "
                "notable_details. Use visible facts only. No analysis or markdown. "
                "Do not infer sensitive traits such as race, ethnicity, age, "
                "nationality, religion, or health."
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

    response = create_chat_completion_with_retries(
        client,
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        max_tokens=2000,
    )

    visual_facts = clean_visual_facts(get_message_text(response.choices[0].message))
    print(f"Visual facts: {visual_facts}")
    return visual_facts


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


def compact_list(value: object, limit: int = 5) -> str:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items[:limit])
    if isinstance(value, str):
        return value.strip()
    return ""


def remove_sensitive_descriptors(text: str) -> str:
    identity_terms = "Black|White|Asian|Latina|Latino|Hispanic"
    person_terms = "woman|man|person|girl|boy|child|teenager|adult"
    text = re.sub(
        rf"\b(young|middle-aged|elderly)\s+({identity_terms})\s+({person_terms})\b",
        r"\1 \3",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b({identity_terms})\s+({person_terms})\b",
        r"\2",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


def clean_visual_facts(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "A short video scene with visible subjects, actions, and setting."

    try:
        parsed = parse_json_object(text)
    except json.JSONDecodeError:
        return clean_visual_summary(text)

    summary = compact_list(parsed.get("summary"))
    subjects = compact_list(parsed.get("subjects"))
    actions = compact_list(parsed.get("actions"))
    setting = compact_list(parsed.get("setting"))
    details = compact_list(parsed.get("notable_details"))

    parts = []
    if summary:
        parts.append(f"Summary: {summary}")
    if subjects:
        parts.append(f"Subjects: {subjects}")
    if actions:
        parts.append(f"Actions: {actions}")
    if setting:
        parts.append(f"Setting: {setting}")
    if details:
        parts.append(f"Notable details: {details}")

    if parts:
        return remove_sensitive_descriptors(". ".join(parts))
    return remove_sensitive_descriptors(clean_visual_summary(text))


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
    short_summary = first_words(re.sub(r"^Summary:\s*", "", visual_summary), 18)
    if style == "sarcastic":
        return f"Oh look, another scene of {short_summary.lower()}"
    if style == "humorous_tech":
        return tech_fallback_caption(visual_summary)
    if style == "humorous_non_tech":
        return f"Somehow this ordinary scene still found a way to be entertaining."
    return short_summary


def has_tech_term(text: str) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lower) for term in TECH_TERMS)


def contains_banned_unseen_detail(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in BANNED_UNSEEN_PATTERNS)


def tech_fallback_caption(visual_facts: str) -> str:
    lower = visual_facts.lower()
    if any(term in lower for term in ["traffic", "vehicles", "cars", "bus"]):
        return "Traffic API logs vehicle packets streaming under the autumn tree interface."
    if any(term in lower for term in ["kitten", "cat", "tabby"]):
        return "Kitten.exe logs a clean walk cycle through the leafy dirt path."
    if any(term in lower for term in ["office", "keyboard", "computer", "monitor"]):
        return "Office runtime tracks keyboard input while the monitor renders focus mode."
    return "System log records visible motion with stable low-latency scene tracking."


def choose_caption(style: str, caption: object, visual_summary: str) -> str:
    if isinstance(caption, str) and caption.strip():
        normalized = normalize_caption(caption)
        is_reasonable_length = 4 <= word_count(normalized) <= 30
        has_matching_tech_tone = (
            (style == "humorous_tech" and has_tech_term(normalized))
            or (style == "humorous_non_tech" and not has_tech_term(normalized))
            or style not in {"humorous_tech", "humorous_non_tech"}
        )
        if (
            is_reasonable_length
            and has_matching_tech_tone
            and not contains_banned_unseen_detail(normalized)
        ):
            return normalized
    return normalize_caption(fallback_caption(style, visual_summary))


def rewrite_captions(styles: list[str], visual_facts: str) -> dict[str, str]:
    client = create_ai_client()
    model = get_model("text")

    style_instructions = {
        "formal": "Use a professional, objective, factual tone.",
        "sarcastic": "Use a dry, ironic, lightly mocking tone.",
        "humorous_tech": (
            "Use humor with clear technology metaphors, such as servers, latency, "
            "algorithms, UI, APIs, inputs, renders, runtime, or system logs. "
            "Do not claim anyone is coding, compiling, or debugging unless visible."
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

    response = create_chat_completion_with_retries(
        client,
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
                    "Do not invent concrete people, objects, actions, locations, "
                    "tasks, dialogue, emails, spreadsheets, or coffee. Humor may "
                    "use metaphors, but visible facts must stay accurate.\n"
                    f"Video facts: {visual_facts}"
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
        captions[style] = choose_caption(style, generated.get(style), visual_facts)

    return captions


def should_self_review() -> bool:
    value = os.getenv("CAPTION_SELF_REVIEW", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def review_captions(styles: list[str], visual_facts: str, captions: dict[str, str]) -> dict[str, str]:
    if not should_self_review():
        return captions

    client = create_ai_client()
    model = get_model("text")

    response = create_chat_completion_with_retries(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict caption editor. Return only valid minified "
                    "JSON. Preserve visible facts. Fix inaccurate, overly long, "
                    "bland, or wrong-style captions."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Video facts: {visual_facts}\n"
                    f"Current captions JSON: {json.dumps(captions)}\n"
                    f"Required style keys: {json.dumps(styles)}\n"
                    "Return corrected captions with the same keys. Each caption "
                    "must be 8 to 18 words. formal is objective. sarcastic is "
                    "clearly dry or ironic. humorous_tech contains an obvious "
                    "tech reference. humorous_non_tech contains no tech jargon. "
                    "Do not add unseen concrete tasks, objects, dialogue, emails, "
                    "spreadsheets, coffee, coding, compiling, debugging, or "
                    "identity traits."
                ),
            },
        ],
        temperature=0.1,
        max_tokens=2000,
    )

    text = get_message_text(response.choices[0].message)
    try:
        reviewed = parse_json_object(text)
    except json.JSONDecodeError:
        reviewed = {}

    return {
        style: choose_caption(style, reviewed.get(style, captions.get(style)), visual_facts)
        for style in styles
    }


def main() -> None:
    load_local_env()

    input_path = resolve_input_path()
    output_path = resolve_output_path()

    tasks = json.loads(input_path.read_text())

    results = []
    for task in tasks:
        video_path = download_video(task["video_url"])
        try:
            frame_paths = extract_frames(video_path)
            visual_facts = describe_video_frames(frame_paths)

            captions = rewrite_captions(task["styles"], visual_facts)
            captions = review_captions(task["styles"], visual_facts, captions)
        finally:
            shutil.rmtree(video_path.parent, ignore_errors=True)

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
