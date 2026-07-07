import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


REQUIRED_STYLES = [
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
]

SAMPLE_TASKS = [
    {
        "task_id": "v1",
        "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
        "styles": REQUIRED_STYLES,
    },
    {
        "task_id": "v2",
        "video_url": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
        "styles": REQUIRED_STYLES,
    },
    {
        "task_id": "v3",
        "video_url": "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
        "styles": REQUIRED_STYLES,
    },
]

BAD_PHRASES = [
    "we need",
    "requested style",
    "style instruction",
    "video facts",
    "return a json",
    "caption should",
    "let me",
]

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


def write_sample_input(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "tasks.json").write_text(
        json.dumps(SAMPLE_TASKS, indent=2),
        encoding="utf-8",
    )


def run_container(image: str, input_dir: Path, output_dir: Path, timeout: int) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    env_args = []
    for name in [
        "AI_PROVIDER",
        "FIREWORKS_API_KEY",
        "FIREWORKS_BASE_URL",
        "FIREWORKS_VISION_MODEL",
        "FIREWORKS_TEXT_MODEL",
        "THEBESTAI_API_KEY",
        "THEBESTAI_BASE_URL",
        "THEBESTAI_VISION_MODEL",
        "THEBESTAI_TEXT_MODEL",
        "PIPELINE_MODE",
        "FRAME_WIDTH",
        "USE_CONTACT_SHEET",
        "FRAME_SELECTION_MODE",
        "FRAME_CANDIDATE_INTERVAL_SECONDS",
        "MAX_CANDIDATE_FRAMES",
        "MAX_VISION_FRAMES",
        "CAPTION_SELF_REVIEW",
    ]:
        value = os.getenv(name)
        if value:
            env_args.extend(["-e", f"{name}={value}"])

    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        *env_args,
        "-v",
        f"{input_dir}:/input",
        "-v",
        f"{output_dir}:/output",
        image,
    ]

    started = time.time()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.time() - started

    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Container exited with code {completed.returncode}")

    return elapsed


def word_count(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def validate_results(input_tasks: list[dict], results: object) -> list[str]:
    failures = []
    if not isinstance(results, list):
        return ["results.json must be a JSON list"]

    expected_ids = [task["task_id"] for task in input_tasks]
    result_ids = [item.get("task_id") for item in results if isinstance(item, dict)]
    if result_ids != expected_ids:
        failures.append(f"task_id order mismatch: expected {expected_ids}, got {result_ids}")

    results_by_id = {
        item.get("task_id"): item
        for item in results
        if isinstance(item, dict)
    }

    for task in input_tasks:
        task_id = task["task_id"]
        item = results_by_id.get(task_id)
        if not item:
            failures.append(f"{task_id}: missing result item")
            continue

        captions = item.get("captions")
        if not isinstance(captions, dict):
            failures.append(f"{task_id}: captions must be an object")
            continue

        for style in task["styles"]:
            caption = captions.get(style)
            if not isinstance(caption, str) or not caption.strip():
                failures.append(f"{task_id}.{style}: missing or empty caption")
                continue

            lower = caption.lower()
            count = word_count(caption)
            if count < 4 or count > 30:
                failures.append(f"{task_id}.{style}: suspicious length ({count} words)")
            if any(phrase in lower for phrase in BAD_PHRASES):
                failures.append(f"{task_id}.{style}: looks like model reasoning leaked")

            has_tech = any(
                re.search(rf"\b{re.escape(term)}\b", lower)
                for term in TECH_TERMS
            )
            has_sarcasm = any(marker in lower for marker in SARCASM_MARKERS)
            if style == "humorous_tech" and not has_tech:
                failures.append(f"{task_id}.{style}: missing clear tech reference")
            if style == "humorous_non_tech" and has_tech:
                failures.append(f"{task_id}.{style}: contains tech jargon")
            if style == "sarcastic" and not has_sarcasm:
                failures.append(f"{task_id}.{style}: sarcasm may be too weak")

    return failures


def main() -> None:
    load_local_env()

    parser = argparse.ArgumentParser(description="Local benchmark for Track 2 container output.")
    parser.add_argument("--image", default="amd-video-caption-agent:test")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    if not (os.getenv("FIREWORKS_API_KEY") or os.getenv("THEBESTAI_API_KEY")):
        raise RuntimeError(
            "Set FIREWORKS_API_KEY or THEBESTAI_API_KEY before running the benchmark."
        )

    benchmark_parent = Path(".benchmark_runs").resolve()
    benchmark_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="caption_benchmark_", dir=benchmark_parent))
    input_dir = temp_root / "input"
    output_dir = temp_root / "output"

    try:
        write_sample_input(input_dir)
        elapsed = run_container(args.image, input_dir, output_dir, args.timeout)

        results_path = output_dir / "results.json"
        if not results_path.exists():
            raise RuntimeError("Container did not write /output/results.json")

        results = json.loads(results_path.read_text(encoding="utf-8"))
        failures = validate_results(SAMPLE_TASKS, results)

        print("\nBenchmark summary")
        print(f"- image: {args.image}")
        print(f"- clips: {len(SAMPLE_TASKS)}")
        print(f"- elapsed: {elapsed:.1f}s")
        print(f"- output: {results_path}")

        if failures:
            print("\nFailures")
            for failure in failures:
                print(f"- {failure}")
            raise SystemExit(1)

        print("\nPASS: output shape and style heuristics look good.")
    finally:
        if args.keep:
            print(f"Kept benchmark files at: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
