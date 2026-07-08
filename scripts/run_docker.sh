#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/noah-wang/amd-video-caption-agent:latest}"

if [[ ! -f ".env" ]]; then
  echo "Missing .env. Copy .env.example to .env and set FIREWORKS_API_KEY." >&2
  exit 1
fi

mkdir -p input output

docker run --rm \
  --platform linux/amd64 \
  --env-file .env \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  "$IMAGE"
