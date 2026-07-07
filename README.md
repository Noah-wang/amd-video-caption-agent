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
5. Use `glm-4v-plus` through the OpenAI-compatible `thebestai` endpoint for visual summarization.
6. Use `deepseek-v4-flash-none` to generate all requested styles in one JSON response.
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
export THEBESTAI_API_KEY="your-key"
```

Optional:

```bash
export THEBESTAI_VISION_MODEL="glm-4v-plus"
export THEBESTAI_TEXT_MODEL="deepseek-v4-flash-none"
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
  -e THEBESTAI_API_KEY="$THEBESTAI_API_KEY" \
  -e THEBESTAI_VISION_MODEL="glm-4v-plus" \
  -e THEBESTAI_TEXT_MODEL="deepseek-v4-flash-none" \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  amd-video-caption-agent:test
```

## Local Benchmark

Run the public sample clips through the container and validate the output shape,
style coverage, obvious reasoning leaks, and simple tone heuristics:

```bash
export THEBESTAI_API_KEY="your-key"
python scripts/local_benchmark.py --image amd-video-caption-agent:test
```

## Submission Build

The judging VM uses `linux/amd64`. On Apple Silicon, build and push with:

```bash
docker buildx build --platform linux/amd64 --tag your-image:latest --push .
```
