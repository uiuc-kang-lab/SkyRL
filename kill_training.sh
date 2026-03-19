#!/bin/bash
# Kills all processes belonging to a SkyRL training run.
# Ray spawns workers in separate process groups, so a plain
# `kill -- -PGID` is not sufficient; we also sweep Ray internals.

PID_FILE="/root/SkyRL/training_eurus_rl_math_qwen3.5-4b_qlora_grpo.pid"

echo "==> Stopping training run: eurus_rl_math_qwen3.5-4b_qlora_grpo"

# 1. Kill the launcher process group (covers non-Ray runs too).
if [ -f "${PID_FILE}" ]; then
  TRAIN_PID=$(cat "${PID_FILE}")
  echo "    Sending SIGKILL to process group ${TRAIN_PID}..."
  kill -9 -- -${TRAIN_PID} 2>/dev/null || true
  rm -f "${PID_FILE}"
else
  echo "    PID file not found (${PID_FILE}), skipping group kill."
fi

# 2. Kill Ray workers and infrastructure by matching known patterns.
for pattern in \
  "ray/_private/workers" \
  "raylet" \
  "plasma_store" \
  "gcs_server" \
  "eurus_rl_math_qwen3.5-4b_qlora_grpo" \
; do
  pkill -9 -f "${pattern}" 2>/dev/null || true
done

# 3. Clean up the Ray temp directory to avoid stale socket errors.
RAY_TMP=$(ls -dt /tmp/ray/session_* 2>/dev/null | head -1)
if [ -n "${RAY_TMP}" ]; then
  echo "    Removing Ray session dir: ${RAY_TMP}"
  rm -rf "${RAY_TMP}"
fi

sleep 1
REMAINING=$(ps aux | grep -E "ray/_private|raylet|plasma|gcs_server|eurus_rl_math_qwen3.5-4b_qlora_grpo" | grep -v grep | wc -l)
echo "==> Done. Remaining related processes: ${REMAINING}"
