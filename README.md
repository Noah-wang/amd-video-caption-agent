# AMD Video Caption Agent

Track 2 submission for the AMD Developer Hackathon. The container reads video
captioning tasks from `/input/tasks.json`, downloads each video, samples frames,
uses a vision model to produce a factual summary, then rewrites that summary into
the requested caption styles.

## Pipeline

1. Read `/input/tasks.json`.
2. Download each `video_url`.
3. Sample evenly spaced frames with `ffmpeg` by default.
4. Optionally enable keyframe mode to sample candidates every second and select representative frames.
5. Resize selected frames to 384px wide and tile them into one contact sheet.
6. Use Fireworks `minimax-m3` in single-pass mode to produce visual facts and captions.
7. Optionally use a slower two-stage or self-review mode for local experiments.
8. Write `/output/results.json`.

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
export FIREWORKS_VISION_MODEL="accounts/fireworks/models/minimax-m3"
export FIREWORKS_TEXT_MODEL="accounts/fireworks/models/gpt-oss-120b"
export PIPELINE_MODE="single_pass"
export CAPTION_SELF_REVIEW="0"
export FRAME_WIDTH="384"
export USE_CONTACT_SHEET="1"
# Optional: adaptive is fastest; keyframes samples candidates every second.
# export FRAME_SELECTION_MODE="adaptive"
# export FRAME_CANDIDATE_INTERVAL_SECONDS="1"
# export MAX_CANDIDATE_FRAMES="120"
# export MAX_VISION_FRAMES="4"
```

`FRAME_SELECTION_MODE=adaptive` is the safer default for the 10-minute judging limit.
For local experiments, `FRAME_SELECTION_MODE=keyframes` with `MAX_VISION_FRAMES=4`
can improve temporal coverage but may be too slow.
`PIPELINE_MODE=single_pass` uses one multimodal call per clip. `two_stage` is often
more structured but roughly doubles model calls.
`FRAME_WIDTH=384` is the faster submission default; `512` or `768` is sharper but slower.
`USE_CONTACT_SHEET=1` sends one tiled image instead of several image inputs, which is much faster.
`CAPTION_SELF_REVIEW=1` can improve tone, but it adds one extra text-model call per clip.

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
  -e FIREWORKS_VISION_MODEL="accounts/fireworks/models/minimax-m3" \
  -e FIREWORKS_TEXT_MODEL="accounts/fireworks/models/gpt-oss-120b" \
  -e PIPELINE_MODE="single_pass" \
  -e CAPTION_SELF_REVIEW="0" \
  -e FRAME_WIDTH="384" \
  -e USE_CONTACT_SHEET="1" \
  -e FRAME_SELECTION_MODE="adaptive" \
  -e FRAME_CANDIDATE_INTERVAL_SECONDS="1" \
  -e MAX_VISION_FRAMES="4" \
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
