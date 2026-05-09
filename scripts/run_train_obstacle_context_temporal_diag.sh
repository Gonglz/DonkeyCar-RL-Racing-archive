#!/usr/bin/env bash
set -u

if [ "$#" -lt 2 ]; then
  echo "usage: $0 DATASET SAVE_DIR [extra train args...]" >&2
  exit 2
fi

DATASET="$1"
SAVE_DIR="$2"
shift 2

mkdir -p "$SAVE_DIR"

/home/longzhao/miniconda3/envs/donkey37/bin/python \
  /home/longzhao/mysim_public/scripts/train_obstacle_context_temporal.py \
  --dataset "$DATASET" \
  --save-dir "$SAVE_DIR" \
  "$@"
STATUS=$?
printf '%s\n' "$STATUS" > "$SAVE_DIR/exit_status.txt"
exit "$STATUS"
