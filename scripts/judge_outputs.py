import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI


DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"
REQUIRED_SCORE_KEYS = [
    "accuracy",
    "style_match",
    "humor",
    "conciseness",
    "no_hallucination",
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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def word_count(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def has_tech_term(text: str) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lower) for term in TECH_TERMS)


def has_sarcasm_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in SARCASM_MARKERS)


def heuristic_scores(style: str, caption: str) -> dict:
    scores = {key: 4 for key in REQUIRED_SCORE_KEYS}
    count = word_count(caption)
    lower = caption.lower()

    if count < 6 or count > 24:
        scores["conciseness"] = 2
    if any(phrase in lower for phrase in ["we need", "caption should", "let me"]):
        scores["no_hallucination"] = 1
        scores["style_match"] = 2
    if style == "humorous_tech" and not has_tech_term(caption):
        scores["style_match"] = 2
        scores["humor"] = 2
    if style == "humorous_non_tech" and has_tech_term(caption):
        scores["style_match"] = 2
    if style == "sarcastic" and not has_sarcasm_marker(caption):
        scores["style_match"] = 2
        scores["humor"] = 2
    if style == "formal" and has_sarcasm_marker(caption):
        scores["style_match"] = 2

    return scores


def create_client() -> OpenAI:
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is required for LLM judging.")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("FIREWORKS_BASE_URL", DEFAULT_FIREWORKS_BASE_URL),
        timeout=60,
    )


def judge_caption_with_llm(
    client: OpenAI,
    model: str,
    visual_facts: str,
    style: str,
    caption: str,
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict video caption competition judge. "
                    "Return only valid minified JSON. Scores are integers 1-5."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Visual facts: {visual_facts or 'Not provided'}\n"
                    f"Requested style: {style}\n"
                    f"Caption: {caption}\n"
                    "Score this caption on accuracy, style_match, humor, "
                    "conciseness, and no_hallucination. Also include one short "
                    "reason and one short improvement suggestion. JSON keys: "
                    "accuracy, style_match, humor, conciseness, no_hallucination, "
                    "reason, improvement."
                ),
            },
        ],
        temperature=0.1,
        max_tokens=500,
    )
    parsed = parse_json_object(response.choices[0].message.content or "")
    for key in REQUIRED_SCORE_KEYS:
        parsed[key] = int(parsed.get(key, 1))
    return parsed


def average_score(scores: dict) -> float:
    return sum(scores[key] for key in REQUIRED_SCORE_KEYS) / len(REQUIRED_SCORE_KEYS)


def facts_by_task(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    data = read_json(path)
    return {
        item.get("task_id"): item.get("visual_facts", "")
        for item in data
        if isinstance(item, dict)
    }


def judge_results(
    tasks: list[dict],
    results: list[dict],
    facts: dict[str, str],
    model: str,
    heuristic_only: bool,
) -> dict:
    client = None if heuristic_only else create_client()
    results_by_id = {item.get("task_id"): item for item in results}
    items = []

    for task in tasks:
        task_id = task["task_id"]
        result = results_by_id.get(task_id, {})
        captions = result.get("captions", {})
        visual_facts = facts.get(task_id, "")

        for style in task.get("styles", []):
            caption = captions.get(style, "")
            if not isinstance(caption, str) or not caption.strip():
                scores = {key: 1 for key in REQUIRED_SCORE_KEYS}
                reason = "Missing caption."
                improvement = "Return a non-empty caption."
            elif heuristic_only:
                scores = heuristic_scores(style, caption)
                reason = "Heuristic style and length check."
                improvement = "Use LLM judging for deeper accuracy checks."
            else:
                judged = judge_caption_with_llm(
                    client,
                    model,
                    visual_facts,
                    style,
                    caption,
                )
                scores = {key: judged[key] for key in REQUIRED_SCORE_KEYS}
                reason = judged.get("reason", "")
                improvement = judged.get("improvement", "")

            items.append(
                {
                    "task_id": task_id,
                    "style": style,
                    "caption": caption,
                    "scores": scores,
                    "average": round(average_score(scores), 3),
                    "reason": reason,
                    "improvement": improvement,
                }
            )

    overall = (
        sum(item["average"] for item in items) / len(items)
        if items
        else 0.0
    )
    return {
        "overall_average": round(overall, 3),
        "items": items,
    }


def main() -> None:
    load_local_env()

    parser = argparse.ArgumentParser(description="Judge generated video captions.")
    parser.add_argument("--input", default="input/tasks.json")
    parser.add_argument("--results", default="output/results.json")
    parser.add_argument("--facts", default="")
    parser.add_argument("--output", default="output/judgement.json")
    parser.add_argument(
        "--model",
        default=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
    )
    parser.add_argument("--heuristic-only", action="store_true")
    args = parser.parse_args()

    tasks = read_json(Path(args.input))
    results = read_json(Path(args.results))
    facts = facts_by_task(Path(args.facts) if args.facts else None)

    judgement = judge_results(
        tasks=tasks,
        results=results,
        facts=facts,
        model=args.model,
        heuristic_only=args.heuristic_only,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(judgement, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Overall average: {judgement['overall_average']:.3f}")
    print(f"Wrote judgement to: {output_path}")


if __name__ == "__main__":
    main()
