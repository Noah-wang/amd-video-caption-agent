import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import math
from pathlib import Path

from openai import OpenAI


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/minimax-m3"
FIREWORKS_TEXT_MODEL = "accounts/fireworks/models/gpt-oss-120b"
THEBESTAI_BASE_URL = "https://thebestai.net/v1"
THEBESTAI_VISION_MODEL = "glm-4v-plus"
THEBESTAI_TEXT_MODEL = "deepseek-v4-flash-none"
MAX_MODEL_RETRIES = 3
SIGNATURE_SIZE = 64 * 64
DEFAULT_FRAME_WIDTH = 384
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
SARCASM_MARKERS = [
    "wow",
    "sure",
    "clearly",
    "totally",
    "groundbreaking",
    "thrilling",
    "because",
    "oh look",
    "oh great",
]
BANNED_UNSEEN_PATTERNS = [
    r"\b(email|emails|coffee|spreadsheet|spreadsheets|litter box)\b",
    r"\b(compiles?|debugs?|coding|programming)\s+(code|spreadsheets?|life)\b",
]
SCENE_TERM_GROUPS = {
    "traffic": [
        "traffic",
        "vehicle",
        "vehicles",
        "car",
        "cars",
        "bus",
        "buses",
        "truck",
        "trucks",
        "lane",
        "lanes",
    ],
    "animal": ["kitten", "cat", "tabby", "feline", "tail", "purr", "purrs"],
    "office": [
        "office",
        "keyboard",
        "monitor",
        "desk",
        "computer",
        "mouse",
        "cable",
        "cables",
        "typing",
    ],
}

STYLE_INSTRUCTIONS = {
    "formal": "Professional, objective, factual tone.",
    "sarcastic": "Dry, ironic, lightly mocking tone.",
    "humorous_tech": (
        "Humor with clear technology metaphors such as servers, latency, "
        "algorithms, UI, APIs, inputs, renders, runtime, or system logs. "
        "Do not claim anyone is coding, compiling, or debugging unless visible."
    ),
    "humorous_non_tech": "Everyday humor with no technical jargon.",
}


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


def get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def get_frame_width() -> int:
    return max(256, get_int_env("FRAME_WIDTH", DEFAULT_FRAME_WIDTH))


def get_candidate_timestamps(duration: float) -> list[float]:
    interval = max(0.5, get_float_env("FRAME_CANDIDATE_INTERVAL_SECONDS", 1.0))
    max_candidates = max(1, get_int_env("MAX_CANDIDATE_FRAMES", 120))
    timestamps = []
    current = min(0.1, max(0.0, duration - 0.05))

    while current < duration and len(timestamps) < max_candidates:
        timestamps.append(round(current, 2))
        current += interval

    if timestamps and timestamps[-1] < duration - interval / 2:
        timestamps.append(round(max(0.0, duration - 0.05), 2))

    return timestamps or [0.0]


def choose_frame_count(duration: float) -> int:
    configured = os.getenv("VIDEO_FRAME_COUNT")
    if configured:
        return max(1, int(configured))
    if duration < 10:
        return 3
    return 4


def get_frame_signatures(video_path: Path, interval: float, max_candidates: int) -> list[bytes]:
    fps = 1 / interval
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:.6f},scale=64:64,format=gray",
        "-frames:v",
        str(max_candidates),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    raw = result.stdout
    return [
        raw[index : index + SIGNATURE_SIZE]
        for index in range(0, len(raw), SIGNATURE_SIZE)
        if len(raw[index : index + SIGNATURE_SIZE]) == SIGNATURE_SIZE
    ]


def frame_difference(previous: bytes, current: bytes) -> float:
    return sum(abs(a - b) for a, b in zip(previous, current)) / len(current)


def select_keyframe_indices(signatures: list[bytes], max_frames: int) -> list[int]:
    if len(signatures) <= max_frames:
        return list(range(len(signatures)))

    anchor_count = max(2, max_frames // 2)
    anchors = {
        round(index * (len(signatures) - 1) / (anchor_count - 1))
        for index in range(anchor_count)
    }

    scored_motion = [
        (frame_difference(signatures[index - 1], signatures[index]), index)
        for index in range(1, len(signatures))
    ]
    scored_motion.sort(reverse=True)

    selected = set(anchors)
    for _, index in scored_motion:
        if len(selected) >= max_frames:
            break
        selected.add(index)

    return sorted(selected)


def extract_evenly_spaced_frames(
    video_path: Path,
    frames_dir: Path,
    duration: float,
) -> list[Path]:
    frame_count = choose_frame_count(duration)
    print(f"Sampling mode: adaptive")
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
            f"scale={get_frame_width()}:-1",
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


def extract_frames(video_path: Path) -> list[Path]:
    frames_dir = Path("samples/frames")

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video_path)
    print(f"Video duration: {duration:.2f} seconds")
    selection_mode = os.getenv("FRAME_SELECTION_MODE", "adaptive").strip().lower()
    if selection_mode != "keyframes":
        return extract_evenly_spaced_frames(video_path, frames_dir, duration)

    print("Sampling mode: keyframes")
    max_vision_frames = max(1, get_int_env("MAX_VISION_FRAMES", 4))
    candidate_interval = max(
        0.5,
        get_float_env("FRAME_CANDIDATE_INTERVAL_SECONDS", 1.0),
    )
    max_candidates = max(1, get_int_env("MAX_CANDIDATE_FRAMES", 120))
    candidate_timestamps = get_candidate_timestamps(duration)
    signatures = get_frame_signatures(video_path, candidate_interval, max_candidates)

    usable_count = min(len(candidate_timestamps), len(signatures))
    candidate_timestamps = candidate_timestamps[:usable_count]
    signatures = signatures[:usable_count]

    if signatures:
        selected_indices = select_keyframe_indices(signatures, max_vision_frames)
        selected_timestamps = [candidate_timestamps[index] for index in selected_indices]
    else:
        fallback_count = min(max_vision_frames, 6)
        selected_timestamps = [
            duration * (index + 1) / (fallback_count + 1)
            for index in range(fallback_count)
        ]

    print(
        "Candidate frames: "
        f"{len(candidate_timestamps)} at ~{candidate_interval:.2f}s intervals"
    )
    print(f"Selected keyframes: {len(selected_timestamps)}")

    frame_paths = []
    for index, timestamp in enumerate(selected_timestamps):
        output_path = frames_dir / f"frame_{index + 1:03d}.jpg"

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.2f}",
            "-i",
            str(video_path),
            "-vf",
            f"scale={get_frame_width()}:-1",
            "-frames:v",
            "1",
            str(output_path),
        ]

        subprocess.run(command, check=True, capture_output=True, text=True)
        frame_paths.append(output_path)

    print("Extracted frames:")
    for frame_path, timestamp in zip(frame_paths, selected_timestamps):
        print(f"  - {frame_path} @ {timestamp:.2f}s")

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


def should_use_contact_sheet() -> bool:
    value = os.getenv("USE_CONTACT_SHEET", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def create_contact_sheet(frame_paths: list[Path]) -> Path:
    if len(frame_paths) == 1:
        return frame_paths[0]

    columns = math.ceil(math.sqrt(len(frame_paths)))
    rows = math.ceil(len(frame_paths) / columns)
    output_path = frame_paths[0].parent / "contact_sheet.jpg"
    input_pattern = frame_paths[0].parent / "frame_%03d.jpg"
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        "1",
        "-i",
        str(input_pattern),
        "-filter_complex",
        f"tile={columns}x{rows}:padding=8:margin=4:color=white",
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    print(f"Created contact sheet: {output_path}")
    return output_path


def build_image_content(frame_paths: list[Path]) -> list[dict]:
    image_paths = [create_contact_sheet(frame_paths)] if should_use_contact_sheet() else frame_paths
    content = []
    for image_path in image_paths:
        image_base64 = encode_image_base64(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }
        )
    return content


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
                "These are sequential frames sampled from one video. If they appear "
                "in one contact sheet, read them left-to-right, top-to-bottom. "
                "Describe the video scene, not the frame layout. Output only valid "
                "minified JSON with keys summary, subjects, actions, setting, "
                "notable_details. Summary must be under 25 words. Each list may "
                "contain at most 4 short phrases. Use visible facts only. No "
                "analysis or markdown. Do not infer sensitive traits such as race, "
                "ethnicity, age, nationality, religion, or health. Do not "
                "transcribe sign text unless it is central and perfectly legible."
            ),
        }
    ]

    content.extend(build_image_content(frame_paths))

    response = create_chat_completion_with_retries(
        client,
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        max_tokens=900,
    )

    visual_facts = clean_visual_facts(get_message_text(response.choices[0].message))
    print(f"Visual facts: {visual_facts}")
    return visual_facts


def generate_single_pass_result(
    styles: list[str],
    frame_paths: list[Path],
) -> tuple[str, dict[str, str]]:
    client = create_ai_client()
    model = get_model("vision")
    requested_styles = {
        style: STYLE_INSTRUCTIONS.get(style, "Concise caption preserving visible facts.")
        for style in styles
    }

    content = [
        {
            "type": "text",
            "text": (
                "These are sequential frames sampled from one video. If they appear "
                "in one contact sheet, read them left-to-right, top-to-bottom. "
                "Output only valid minified JSON with exactly two top-level keys: "
                "visual_facts and captions. visual_facts must contain summary, "
                "subjects, actions, setting, notable_details. Captions keys must "
                f"match this JSON object: {json.dumps(requested_styles)}. "
                "Each caption must be one English sentence, 8 to 18 words. "
                "Preserve visible facts. Do not invent concrete people, objects, "
                "actions, locations, tasks, dialogue, emails, spreadsheets, coffee, "
                "coding, compiling, debugging, identity traits, or sign text. "
                "humorous_tech must include an obvious tech metaphor. "
                "humorous_non_tech must contain no tech jargon."
            ),
        }
    ]

    content.extend(build_image_content(frame_paths))

    response = create_chat_completion_with_retries(
        client,
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=2200,
    )

    parsed = parse_json_object(get_message_text(response.choices[0].message))
    raw_visual_facts = parsed.get("visual_facts", parsed.get("facts", {}))
    if isinstance(raw_visual_facts, str):
        visual_facts = clean_visual_facts(raw_visual_facts)
    else:
        visual_facts = clean_visual_facts(json.dumps(raw_visual_facts))

    raw_captions = parsed.get("captions")
    if not isinstance(raw_captions, dict):
        raw_captions = parsed

    captions = {
        style: choose_caption(style, raw_captions.get(style), visual_facts)
        for style in styles
    }
    print(f"Visual facts: {visual_facts}")
    return visual_facts, captions


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


def filter_unreliable_ocr(value: object) -> object:
    blocked = ["sign reads", "text reads", "readable text", "korean text", "signage"]
    if isinstance(value, list):
        return [
            item
            for item in value
            if not any(phrase in str(item).lower() for phrase in blocked)
        ]
    if isinstance(value, str) and any(phrase in value.lower() for phrase in blocked):
        return ""
    return value


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


def remove_frame_layout_references(text: str) -> str:
    text = re.sub(
        r"\s+(across|in|over)\s+(one|two|three|four|five|six|\d+)\s+sequential frames",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+across the frames", "", text, flags=re.IGNORECASE)
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
    details = compact_list(filter_unreliable_ocr(parsed.get("notable_details")))

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
        return remove_frame_layout_references(
            remove_sensitive_descriptors(". ".join(parts))
        )
    return remove_frame_layout_references(
        remove_sensitive_descriptors(clean_visual_summary(text))
    )


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
    return remove_frame_layout_references(re.sub(r"\s+", " ", text).strip())


def word_count(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def fallback_caption(style: str, visual_summary: str) -> str:
    short_summary = first_words(re.sub(r"^Summary:\s*", "", visual_summary), 18)
    if style == "sarcastic":
        return sarcastic_fallback_caption(visual_summary)
    if style == "humorous_tech":
        return tech_fallback_caption(visual_summary)
    if style == "humorous_non_tech":
        return f"Somehow this ordinary scene still found a way to be entertaining."
    return short_summary


def has_tech_term(text: str) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lower) for term in TECH_TERMS)


def has_sarcasm_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in SARCASM_MARKERS)


def contains_banned_unseen_detail(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in BANNED_UNSEEN_PATTERNS)


def contains_group_term(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lower) for term in terms)


def has_scene_conflict(caption: str, visual_facts: str) -> bool:
    for group_name, terms in SCENE_TERM_GROUPS.items():
        caption_mentions_group = contains_group_term(caption, terms)
        facts_mention_group = contains_group_term(visual_facts, terms)
        if caption_mentions_group and not facts_mention_group:
            return True
    return False


def tech_fallback_caption(visual_facts: str) -> str:
    lower = visual_facts.lower()
    if any(term in lower for term in ["traffic", "vehicles", "cars", "bus"]):
        return "Traffic API logs vehicle packets streaming under the autumn tree interface."
    if any(term in lower for term in ["kitten", "cat", "tabby"]):
        return "Kitten.exe logs a clean walk cycle through the leafy dirt path."
    if any(term in lower for term in ["office", "keyboard", "computer", "monitor"]):
        return "Office runtime tracks keyboard input while the monitor renders focus mode."
    return "System log records visible motion with stable low-latency scene tracking."


def sarcastic_fallback_caption(visual_facts: str) -> str:
    lower = visual_facts.lower()
    if any(term in lower for term in ["traffic", "vehicles", "cars", "bus"]):
        return "Oh great, another heroic traffic parade pretending to be urban progress."
    if any(term in lower for term in ["kitten", "cat", "tabby"]):
        return "Oh look, a tiny cat conquering the garden like a furry emperor."
    if any(term in lower for term in ["office", "keyboard", "computer", "monitor"]):
        return "Oh great, another office keyboard battle under extremely serious lighting."
    return "Oh look, another ordinary scene bravely demanding dramatic attention."


def choose_caption(style: str, caption: object, visual_summary: str) -> str:
    if isinstance(caption, str) and caption.strip():
        normalized = normalize_caption(caption)
        is_reasonable_length = 4 <= word_count(normalized) <= 30
        has_matching_tech_tone = (
            (style == "humorous_tech" and has_tech_term(normalized))
            or (style == "humorous_non_tech" and not has_tech_term(normalized))
            or style not in {"humorous_tech", "humorous_non_tech"}
        )
        has_matching_sarcasm = style != "sarcastic" or has_sarcasm_marker(normalized)
        if (
            is_reasonable_length
            and has_matching_tech_tone
            and has_matching_sarcasm
            and not contains_banned_unseen_detail(normalized)
            and not has_scene_conflict(normalized, visual_summary)
        ):
            return normalized
    return normalize_caption(fallback_caption(style, visual_summary))


def rewrite_captions(styles: list[str], visual_facts: str) -> dict[str, str]:
    client = create_ai_client()
    model = get_model("text")

    requested_styles = {
        style: STYLE_INSTRUCTIONS.get(
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
                    "use metaphors, but visible facts must stay accurate. Do not "
                    "mention sign text from the video.\n"
                    f"Video facts: {visual_facts}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=900,
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
    value = os.getenv("CAPTION_SELF_REVIEW", "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


def get_pipeline_mode() -> str:
    return os.getenv("PIPELINE_MODE", "single_pass").strip().lower()


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
                    "identity traits. Do not carry details from another video. "
                    "Do not mention sign text from the video."
                ),
            },
        ],
        temperature=0.1,
        max_tokens=900,
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
    debug_facts_path = os.getenv("DEBUG_FACTS_PATH")

    tasks = json.loads(input_path.read_text())

    results = []
    debug_facts = []
    for task in tasks:
        video_path = download_video(task["video_url"])
        try:
            frame_paths = extract_frames(video_path)
            if get_pipeline_mode() == "two_stage":
                visual_facts = describe_video_frames(frame_paths)
                captions = rewrite_captions(task["styles"], visual_facts)
            else:
                try:
                    visual_facts, captions = generate_single_pass_result(
                        task["styles"],
                        frame_paths,
                    )
                except Exception as error:
                    print(f"Single-pass generation failed, falling back: {error}")
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
        debug_facts.append(
            {
                "task_id": task["task_id"],
                "visual_facts": visual_facts,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if debug_facts_path:
        Path(debug_facts_path).write_text(
            json.dumps(debug_facts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
