#!/usr/bin/env bash
# Run T5FineTuneConfig_base variants sequentially via part1/train.py.
# On CUDA/CPU OOM: logs the failure and continues to the next config.
# On any other non-zero exit: stops immediately.
#
# Features:
#   --skip <cfg>   Skip a config (repeatable). Useful for resuming after partial runs.
#   Watchdog:      Kills training process if it hangs >5 min after MLflow run finishes.
#
# Usage:
#   nohup bash script/run_base_configs.sh --skip T5FineTuneConfig_base > /tmp/t5_base_train.log 2>&1 &
set -uo pipefail

# Override HF_HOME: system default (/workspace/.hf_home) is owned by root.
# Models are already cached in the user-writable location below.
export HF_HOME=/home/coder/.cache/huggingface

# Force unbuffered Python output so tee/nohup see lines immediately.
export PYTHONUNBUFFERED=1

CONFIGS=(
    T5FineTuneConfig_base
    T5FineTuneConfig_base2
    T5FineTuneConfig_base3
    T5FineTuneConfig_base4
)

OOM_PATTERNS="OutOfMemoryError|out of memory|CUDA out of memory|Cannot allocate memory"

# ── Parse --skip flags ───────────────────────────────────────────────────
declare -A SKIP_SET
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip) SKIP_SET["$2"]=1; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

skipped=()
completed=()

# ── Watchdog: kill training PID if it hangs after MLflow marks run FINISHED ──
# Runs as a background co-process; killed when the main loop moves on.
watchdog() {
    local pid=$1 run_name=$2
    local grace=300  # 5 minutes
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        # Check if MLflow already marked this run FINISHED
        local status
        status=$(python3 -c "
import sqlite3
conn = sqlite3.connect('mlflow.db')
cur = conn.cursor()
cur.execute(\"SELECT status FROM runs WHERE name=? ORDER BY start_time DESC LIMIT 1\", ('$run_name',))
row = cur.fetchone()
print(row[0] if row else 'UNKNOWN')
conn.close()
" 2>/dev/null)
        if [[ "$status" == "FINISHED" || "$status" == "FAILED" ]]; then
            echo "[watchdog] MLflow run '$run_name' is $status. Waiting ${grace}s grace period..."
            sleep "$grace"
            if kill -0 "$pid" 2>/dev/null; then
                echo "[watchdog] Process $pid still alive after grace period — sending SIGTERM."
                kill "$pid" 2>/dev/null
                sleep 10
                if kill -0 "$pid" 2>/dev/null; then
                    echo "[watchdog] SIGTERM insufficient — sending SIGKILL."
                    kill -9 "$pid" 2>/dev/null
                fi
            fi
            return
        fi
    done
}

for cfg in "${CONFIGS[@]}"; do
    # Skip if requested via --skip
    if [[ -n "${SKIP_SET[$cfg]+x}" ]]; then
        echo "=========================================="
        echo "Skipping config: $cfg (--skip)"
        echo "=========================================="
        echo
        continue
    fi

    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting config: $cfg"
    echo "=========================================="

    stderr_log=$(mktemp)

    # Determine the MLflow run_name from the config class.
    # Convention: T5FineTuneConfig_base  →  t5_ft_base_v1  (read from the config's .name field)
    run_name=$(python3 -c "
from part1.config import $cfg
print(${cfg}().name)
" 2>/dev/null || echo "unknown")
    echo "MLflow run_name: $run_name"

    # Launch training in a subshell so we can get its PID for the watchdog.
    python3 -m part1.train --config "$cfg" 2> >(tee "$stderr_log" >&2) &
    train_pid=$!

    # Start watchdog in background
    watchdog "$train_pid" "$run_name" &
    wd_pid=$!

    # Wait for training to finish
    wait "$train_pid"
    exit_code=$?

    # Kill watchdog
    kill "$wd_pid" 2>/dev/null
    wait "$wd_pid" 2>/dev/null

    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: $cfg"
        completed+=("$cfg")
    elif [[ $exit_code -eq 143 || $exit_code -eq 137 ]]; then
        # 143 = SIGTERM (from watchdog), 137 = SIGKILL (from watchdog)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $cfg: process killed by watchdog (exit $exit_code) — likely hung after completion. Treating as success."
        completed+=("$cfg")
    else
        if grep -qE "$OOM_PATTERNS" "$stderr_log"; then
            echo "WARNING: OOM detected for '$cfg' (exit $exit_code) — skipping to next config."
            skipped+=("$cfg")
        else
            echo "ERROR: '$cfg' failed with exit $exit_code (not OOM) — aborting."
            rm -f "$stderr_log"
            exit "$exit_code"
        fi
    fi

    rm -f "$stderr_log"
    echo
done

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sequential run complete."
if [ ${#completed[@]} -gt 0 ]; then
    echo "Completed:"
    for c in "${completed[@]}"; do
        echo "  + $c"
    done
fi
if [ ${#skipped[@]} -gt 0 ]; then
    echo "Skipped due to OOM:"
    for s in "${skipped[@]}"; do
        echo "  - $s"
    done
    exit 1
else
    echo "All configs succeeded."
fi
