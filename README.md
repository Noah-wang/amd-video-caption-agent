# AMD Video Caption Agent

Track 2 submission for the AMD Developer Hackathon. The container reads video
captioning tasks from `/input/tasks.json`, downloads each video, samples frames,
uses a vision model to produce a factual summary, then rewrites that summary into
the requested caption styles.

## Pipeline

1. Read `/input/tasks.json`.
2. Download each `video_url`.
3. Sample 4 evenly spaced frames with `ffmpeg`.
4. Resize frames to 768px wide to reduce image-token cost.
5. Use Fireworks `qwen2p5-vl-32b-instruct` for visual summarization.
6. Use Fireworks `qwen2p5-32b-instruct` to generate all requested styles in one JSON response.
7. Write `/output/results.json`.

## Input

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

## Output

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

## Environment

Required:

```bash
export FIREWORKS_API_KEY="your-key"
```

Optional:

```bash
export AI_PROVIDER="fireworks"
export FIREWORKS_VISION_MODEL="accounts/fireworks/models/qwen2p5-vl-32b-instruct"
export FIREWORKS_TEXT_MODEL="accounts/fireworks/models/qwen2p5-32b-instruct"
```

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app/main.py
cat output/results.json
```

## Docker Run

```bash
docker build --platform linux/amd64 -t amd-video-caption-agent:test .
docker run --rm \
  --platform linux/amd64 \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e AI_PROVIDER="fireworks" \
  -e FIREWORKS_VISION_MODEL="accounts/fireworks/models/qwen2p5-vl-32b-instruct" \
  -e FIREWORKS_TEXT_MODEL="accounts/fireworks/models/qwen2p5-32b-instruct" \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  amd-video-caption-agent:test
```

## Local Benchmark

Run the public sample clips through the container and validate the output shape,
style coverage, obvious reasoning leaks, and simple tone heuristics:

```bash
export FIREWORKS_API_KEY="your-key"
python scripts/local_benchmark.py --image amd-video-caption-agent:test
```

## Submission Build

The judging VM uses `linux/amd64`. On Apple Silicon, build and push with:

```bash
docker buildx build --platform linux/amd64 --tag your-image:latest --push .
```
