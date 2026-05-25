#!/usr/bin/env bash
set -u

cd /home/jetson/mycar || exit 1
. /home/jetson/env/bin/activate
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

STAMP=${1:-$(date +%Y%m%d_%H%M%S)}
BASE=/home/jetson/mycar/monitor_logs/v17_final_20min_shadow_${STAMP}
RUN=trt_shadow_20min_final
RUN_DIR="$BASE/$RUN"
MODEL=/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip
ENGINE=/home/jetson/mycar/models/v17_actor_fp16.engine
META=/home/jetson/mycar/models/v17_actor_export.json

mkdir -p "$RUN_DIR"
{
  echo "base_dir=$BASE"
  echo "run=$RUN"
  echo "started_at=$(date -Iseconds)"
  echo "model=$MODEL"
  echo "engine=$ENGINE"
  echo "metadata=$META"
  echo "control_mode=shadow"
  echo "duration_sec=1200"
} > "$RUN_DIR/run_context.txt"

cat > "$RUN_DIR/command.txt" <<CMD
python runtime_monitor.py drive --model "$MODEL" --type v17 --js --control-mode shadow --shadow-duration 1200 --log-dir "$RUN_DIR" --run-label "$RUN" --track-condition final_endpoint_deployment_freeze --shadow-engine "$ENGINE" --shadow-engine-metadata "$META" --force-recording
CMD

timeout 1500s python runtime_monitor.py drive \
  --model "$MODEL" \
  --type v17 \
  --js \
  --control-mode shadow \
  --shadow-duration 1200 \
  --log-dir "$RUN_DIR" \
  --run-label "$RUN" \
  --track-condition final_endpoint_deployment_freeze \
  --shadow-engine "$ENGINE" \
  --shadow-engine-metadata "$META" \
  --force-recording \
  > "$RUN_DIR/runtime.log" 2>&1
rc=$?
echo "$rc" > "$RUN_DIR/exit_code.txt"

if ls "$RUN_DIR"/run_*.csv >/dev/null 2>&1; then
  python tools/summarize_shadow_run.py \
    --csv "$RUN_DIR"/run_*.csv \
    --model-path "$MODEL" \
    --out "$RUN_DIR/summary.json" \
    --log-copy "$RUN_DIR/jetson_shadow_log.csv" \
    > "$RUN_DIR/summarize.log" 2>&1
fi

{
  echo "completed_at=$(date -Iseconds)"
  echo "exit_code=$rc"
} >> "$RUN_DIR/run_context.txt"

echo "DONE rc=$rc" > "$RUN_DIR/DONE"
exit "$rc"
