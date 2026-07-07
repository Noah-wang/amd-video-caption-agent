import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from judge_outputs import judge_results, facts_by_task
from local_benchmark import (
    SAMPLE_TASKS,
    load_local_env,
    run_container,
    validate_results,
    write_sample_input,
)


CONFIGS = [
    {
        "name": "fast_single_pass_384",
        "env": {
            "PIPELINE_MODE": "single_pass",
            "CAPTION_SELF_REVIEW": "0",
            "FRAME_SELECTION_MODE": "adaptive",
            "FRAME_WIDTH": "384",
            "USE_CONTACT_SHEET": "1",
        },
    },
    {
        "name": "sharper_single_pass_512",
        "env": {
            "PIPELINE_MODE": "single_pass",
            "CAPTION_SELF_REVIEW": "0",
            "FRAME_SELECTION_MODE": "adaptive",
            "FRAME_WIDTH": "512",
            "USE_CONTACT_SHEET": "1",
        },
    },
    {
        "name": "keyframes_single_pass_384",
        "env": {
            "PIPELINE_MODE": "single_pass",
            "CAPTION_SELF_REVIEW": "0",
            "FRAME_SELECTION_MODE": "keyframes",
            "FRAME_CANDIDATE_INTERVAL_SECONDS": "1",
            "MAX_CANDIDATE_FRAMES": "120",
            "MAX_VISION_FRAMES": "4",
            "FRAME_WIDTH": "384",
            "USE_CONTACT_SHEET": "1",
        },
    },
    {
        "name": "two_stage_384",
        "env": {
            "PIPELINE_MODE": "two_stage",
            "CAPTION_SELF_REVIEW": "0",
            "FRAME_SELECTION_MODE": "adaptive",
            "FRAME_WIDTH": "384",
            "USE_CONTACT_SHEET": "1",
        },
    },
]


def apply_env(overrides: dict[str, str]) -> dict[str, str | None]:
    previous = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_one_config(
    image: str,
    config: dict,
    root: Path,
    timeout: int,
    judge_model: str,
    heuristic_only: bool,
) -> dict:
    run_dir = root / config["name"]
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    write_sample_input(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(config["env"])
    env["DEBUG_FACTS_PATH"] = "/output/debug_facts.json"
    previous_env = apply_env(env)
    try:
        elapsed = run_container(image, input_dir, output_dir, timeout)
    finally:
        restore_env(previous_env)

    results_path = output_dir / "results.json"
    facts_path = output_dir / "debug_facts.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    failures = validate_results(SAMPLE_TASKS, results)
    judgement = judge_results(
        tasks=SAMPLE_TASKS,
        results=results,
        facts=facts_by_task(facts_path),
        model=judge_model,
        heuristic_only=heuristic_only,
    )

    score = judgement["overall_average"]
    score_per_minute = score / max(elapsed / 60, 0.001)
    summary = {
        "name": config["name"],
        "env": config["env"],
        "elapsed_seconds": round(elapsed, 1),
        "validation_failures": failures,
        "judge_average": score,
        "score_per_minute": round(score_per_minute, 3),
        "results_path": str(results_path),
        "facts_path": str(facts_path),
    }
    (output_dir / "judgement.json").write_text(
        json.dumps(judgement, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    load_local_env()

    parser = argparse.ArgumentParser(description="Sweep caption pipeline configs.")
    parser.add_argument("--image", default="amd-video-caption-agent:test")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--heuristic-only", action="store_true")
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "accounts/fireworks/models/gpt-oss-120b"),
    )
    parser.add_argument("--output", default="output/sweep_summary.json")
    args = parser.parse_args()

    if not os.getenv("FIREWORKS_API_KEY"):
        raise RuntimeError("FIREWORKS_API_KEY is required to run config sweeps.")

    sweep_parent = Path(".sweep_runs").resolve()
    sweep_parent.mkdir(parents=True, exist_ok=True)
    sweep_root = Path(tempfile.mkdtemp(prefix="caption_sweep_", dir=sweep_parent))
    selected_configs = CONFIGS[: args.limit]
    summaries = []

    try:
        for config in selected_configs:
            print(f"\n=== Running {config['name']} ===")
            try:
                summary = run_one_config(
                    image=args.image,
                    config=config,
                    root=sweep_root,
                    timeout=args.timeout,
                    judge_model=args.judge_model,
                    heuristic_only=args.heuristic_only,
                )
            except Exception as error:
                summary = {
                    "name": config["name"],
                    "env": config["env"],
                    "error": str(error),
                }
            summaries.append(summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))

        ranked = sorted(
            summaries,
            key=lambda item: (
                item.get("validation_failures") == [],
                item.get("judge_average", 0),
                item.get("score_per_minute", 0),
            ),
            reverse=True,
        )
        output = {
            "sweep_root": str(sweep_root),
            "ranked": ranked,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote sweep summary to: {output_path}")
    finally:
        if args.keep:
            print(f"Kept sweep files at: {sweep_root}")
        else:
            shutil.rmtree(sweep_root, ignore_errors=True)


if __name__ == "__main__":
    main()
