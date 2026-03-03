#!/usr/bin/env bash
# Monitor T5-base training runs. Detect issues, attempt fixes, and log to doc file.
# Designed to run persistently via nohup even after SSH disconnect.
#
# Usage:
#   nohup bash script/monitor_training.sh > /tmp/monitor_training.log 2>&1 &
set -uo pipefail

DOC_FILE="/home/coder/nlp_Assignment3/doc/t5-base-training-run.md"
TRAIN_LOG="/tmp/t5_base_train_r2.log"
MONITOR_LOG="/tmp/monitor_training.log"
PROJECT_DIR="/home/coder/nlp_Assignment3"
MLFLOW_DB="$PROJECT_DIR/mlflow.db"
OUTPUT_DIR="$PROJECT_DIR/output"
CHECK_INTERVAL=120  # seconds between checks
ISSUE_COUNTER_FILE="/tmp/monitor_issue_counter"
LAST_EPOCH_FILE="/tmp/monitor_last_epoch"
STALL_THRESHOLD=900  # 15 minutes with no new epoch = stalled

export HF_HOME=/home/coder/.cache/huggingface
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"

# Initialize counters
if [[ ! -f "$ISSUE_COUNTER_FILE" ]]; then
    # Read existing issue count from doc file
    existing=$(grep -c '^### Issue' "$DOC_FILE" 2>/dev/null || echo 0)
    echo "$existing" > "$ISSUE_COUNTER_FILE"
fi
echo "$(date -Iseconds)" > "$LAST_EPOCH_FILE"

log() {
    echo "[monitor $(date '+%Y-%m-%d %H:%M:%S')] $*"
}

next_issue_num() {
    local n
    n=$(cat "$ISSUE_COUNTER_FILE")
    n=$((n + 1))
    echo "$n" > "$ISSUE_COUNTER_FILE"
    echo "$n"
}

# Append an issue section to the doc file
append_issue() {
    local title="$1"
    local error_msg="$2"
    local cause="$3"
    local fix="$4"
    local num
    num=$(next_issue_num)
    cat >> "$DOC_FILE" <<EOF

### Issue $num: $title
- **Time**: $(date '+%Y-%m-%d %H:%M:%S UTC')
- **Error**: \`$error_msg\`
- **Cause**: $cause
- **Fix**: $fix
EOF
    log "Documented Issue $num: $title"
}

# Append a note (non-issue) to the doc file
append_note() {
    local msg="$1"
    # Append under Run 2 section
    cat >> "$DOC_FILE" <<EOF

> **[$(date '+%Y-%m-%d %H:%M:%S')]** $msg
EOF
    log "Note: $msg"
}

# Update results table in doc for a given config
update_results_table() {
    local config_name="$1"
    local f1="$2"
    local em="$3"
    local sql_em="$4"
    local epochs="$5"
    local wall_clock="$6"

    # Use python to update the markdown table row
    python3 -c "
import re, sys

config_name = '$config_name'
f1, em, sql_em = '$f1', '$em', '$sql_em'
epochs, wall_clock = '$epochs', '$wall_clock'

with open('$DOC_FILE', 'r') as f:
    content = f.read()

# Find the row for this config and update it
pattern = r'(\| \`' + re.escape(config_name) + r'\`\s*\|).*'
replacement = f'| \`{config_name}\` | **{f1}** | {em} | {sql_em} | {epochs} | {wall_clock} |'
new_content = re.sub(pattern, replacement, content)

if new_content != content:
    with open('$DOC_FILE', 'w') as f:
        f.write(new_content)
    print(f'Updated results row for {config_name}')
else:
    print(f'No row found for {config_name} or already up to date')
" 2>&1
}

# Get the latest epoch info from training log
get_latest_epoch() {
    cat "$TRAIN_LOG" 2>/dev/null | tr '\r' '\n' | grep -oP 'Epoch \d+:.*' | tail -1
}

# Get MLflow run status for a given run name
get_mlflow_status() {
    local run_name="$1"
    python3 -c "
import sqlite3
conn = sqlite3.connect('$MLFLOW_DB')
cur = conn.cursor()
cur.execute(\"SELECT status FROM runs WHERE name=? ORDER BY start_time DESC LIMIT 1\", ('$run_name',))
row = cur.fetchone()
print(row[0] if row else 'NOT_FOUND')
conn.close()
" 2>/dev/null
}

# Get metrics from MLflow for a given run name
get_mlflow_best_metrics() {
    local run_name="$1"
    python3 -c "
import sqlite3
conn = sqlite3.connect('$MLFLOW_DB')
cur = conn.cursor()
# Get run_uuid
cur.execute(\"SELECT run_uuid FROM runs WHERE name=? ORDER BY start_time DESC LIMIT 1\", ('$run_name',))
row = cur.fetchone()
if not row:
    print('NOT_FOUND')
else:
    run_id = row[0]
    metrics = {}
    for key in ['best_record_f1', 'best_record_em', 'best_sql_em', 'best_epoch', 'epoch']:
        cur.execute(\"SELECT value FROM latest_metrics WHERE run_uuid=? AND key=?\", (run_id, key))
        r = cur.fetchone()
        metrics[key] = r[0] if r else 'N/A'
    print(f\"{metrics.get('best_record_f1','N/A')}|{metrics.get('best_record_em','N/A')}|{metrics.get('best_sql_em','N/A')}|{metrics.get('epoch','N/A')}|{metrics.get('best_epoch','N/A')}\")
conn.close()
" 2>/dev/null
}

# Check if training process is alive
is_training_alive() {
    pgrep -f "part1.train" > /dev/null 2>&1
}

# Check if the shell script is alive
is_script_alive() {
    pgrep -f "run_base_configs" > /dev/null 2>&1
}

# Check GPU health
check_gpu() {
    nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null
}

# Check disk space
check_disk() {
    df -h / | awk 'NR==2{print $5}' | tr -d '%'
}

# ── Checks for common issues ──

check_oom() {
    # Check if training crashed with OOM since last check
    local recent_errors
    recent_errors=$(cat "$TRAIN_LOG" 2>/dev/null | tr '\r' '\n' | grep -iE '(OutOfMemoryError|CUDA out of memory|Cannot allocate memory|RuntimeError.*memory)' | tail -3)
    if [[ -n "$recent_errors" ]]; then
        echo "$recent_errors"
        return 0
    fi
    return 1
}

check_nan_loss() {
    local recent_nan
    recent_nan=$(cat "$TRAIN_LOG" 2>/dev/null | tr '\r' '\n' | grep -iE '(nan|inf)' | grep -i 'loss' | tail -3)
    if [[ -n "$recent_nan" ]]; then
        echo "$recent_nan"
        return 0
    fi
    return 1
}

check_gpu_temp() {
    local gpu_info
    gpu_info=$(check_gpu)
    if [[ -z "$gpu_info" ]]; then
        return 1
    fi
    local temp
    temp=$(echo "$gpu_info" | cut -d',' -f1 | tr -d ' ')
    if [[ "$temp" -gt 85 ]]; then
        echo "GPU temperature: ${temp}C"
        return 0
    fi
    return 1
}

check_disk_full() {
    local usage
    usage=$(check_disk)
    if [[ "$usage" -gt 95 ]]; then
        echo "Disk usage: ${usage}%"
        return 0
    fi
    return 1
}

# Track which configs have been seen as FINISHED (to avoid duplicate doc updates)
declare -A SEEN_FINISHED

log "=========================================="
log "Monitor started."
log "Train log: $TRAIN_LOG"
log "Doc file: $DOC_FILE"
log "Check interval: ${CHECK_INTERVAL}s"
log "=========================================="

# ── Main monitoring loop ──
prev_epoch_line=""
stall_start=""

while true; do
    # 1) Check if the whole sequential run is done
    if ! is_script_alive && ! is_training_alive; then
        log "Neither run_base_configs.sh nor training process found. Run appears complete."

        # Final update: check all configs in MLflow
        for cfg_name in t5_ft_base_v2 t5_ft_base_v3 t5_ft_base_v4; do
            status=$(get_mlflow_status "$cfg_name")
            if [[ "$status" == "FINISHED" && -z "${SEEN_FINISHED[$cfg_name]+x}" ]]; then
                SEEN_FINISHED["$cfg_name"]=1
                metrics=$(get_mlflow_best_metrics "$cfg_name")
                if [[ "$metrics" != "NOT_FOUND" ]]; then
                    IFS='|' read -r f1 em sql_em epoch best_ep <<< "$metrics"
                    update_results_table "$cfg_name" "$f1" "$em" "$sql_em" "${epoch} (best@${best_ep})" "—"
                fi
            fi
        done

        append_note "Monitor: sequential run completed. All configs processed."
        log "Exiting monitor."
        break
    fi

    # 2) Check for OOM
    if oom_msg=$(check_oom); then
        log "OOM detected: $oom_msg"
        append_issue "CUDA OOM during training" \
            "$(echo "$oom_msg" | head -1)" \
            "GPU memory exhausted during forward/backward pass or evaluation." \
            "The run_base_configs.sh script will skip this config and continue to the next. No manual intervention needed."
    fi

    # 3) Check for NaN loss
    if nan_msg=$(check_nan_loss); then
        log "NaN loss detected: $nan_msg"
        append_issue "NaN loss detected" \
            "$(echo "$nan_msg" | head -1)" \
            "Numerical instability, possibly from too-high learning rate or gradient explosion." \
            "Training will likely fail and early-stop. Consider lowering learning_rate or increasing grad_clip_norm for future runs."
    fi

    # 4) Check GPU temperature
    if temp_msg=$(check_gpu_temp); then
        log "HIGH GPU TEMP: $temp_msg"
        append_issue "GPU overheating" \
            "$temp_msg" \
            "Sustained compute load pushing GPU beyond safe operating temperature." \
            "Monitor closely. If persistent, consider reducing batch_size or adding cooling breaks."
    fi

    # 5) Check disk space
    if disk_msg=$(check_disk_full); then
        log "DISK NEARLY FULL: $disk_msg"
        # Attempt fix: clean old checkpoints
        log "Attempting to free space by removing old intermediate checkpoints..."
        freed=0
        for ckpt_dir in "$OUTPUT_DIR"/*/checkpoints; do
            if [[ -d "$ckpt_dir" ]]; then
                count=$(find "$ckpt_dir" -name "*.pt" -o -name "*.bin" | wc -l)
                if [[ $count -gt 1 ]]; then
                    # Keep only the latest checkpoint
                    find "$ckpt_dir" -name "*.pt" -o -name "*.bin" | sort | head -n -1 | xargs rm -f
                    freed=1
                fi
            fi
        done
        fix_msg="Cleaned old intermediate checkpoints to free disk space."
        [[ $freed -eq 0 ]] && fix_msg="No old checkpoints to clean. Manual intervention may be needed."
        append_issue "Disk space critically low" \
            "$disk_msg" \
            "Accumulated checkpoints and model files consuming disk." \
            "$fix_msg"
    fi

    # 6) Check for stalled training (no new epoch in STALL_THRESHOLD seconds)
    curr_epoch_line=$(get_latest_epoch)
    if [[ "$curr_epoch_line" != "$prev_epoch_line" ]]; then
        prev_epoch_line="$curr_epoch_line"
        stall_start=""
        echo "$(date -Iseconds)" > "$LAST_EPOCH_FILE"
    elif is_training_alive; then
        if [[ -z "$stall_start" ]]; then
            stall_start=$(date +%s)
        else
            now=$(date +%s)
            elapsed=$((now - stall_start))
            if [[ $elapsed -gt $STALL_THRESHOLD ]]; then
                log "Training appears stalled for ${elapsed}s"
                # Check if MLflow already finished (hung process scenario)
                for cfg_name in t5_ft_base_v2 t5_ft_base_v3 t5_ft_base_v4; do
                    status=$(get_mlflow_status "$cfg_name")
                    if [[ "$status" == "RUNNING" ]]; then
                        log "MLflow run '$cfg_name' still RUNNING — possible genuine stall."
                    fi
                done
                append_note "Monitor: training appears stalled (no new epoch for ${elapsed}s). Watchdog in run_base_configs.sh will handle hung processes if MLflow run completes."
                stall_start=""  # Reset to avoid spamming
            fi
        fi
    fi

    # 7) Track config completions and update results table
    for cfg_name in t5_ft_base_v2 t5_ft_base_v3 t5_ft_base_v4; do
        if [[ -n "${SEEN_FINISHED[$cfg_name]+x}" ]]; then
            continue
        fi
        status=$(get_mlflow_status "$cfg_name")
        if [[ "$status" == "FINISHED" || "$status" == "FAILED" ]]; then
            SEEN_FINISHED["$cfg_name"]=1
            log "Config '$cfg_name' finished with status: $status"

            if [[ "$status" == "FINISHED" ]]; then
                metrics=$(get_mlflow_best_metrics "$cfg_name")
                if [[ "$metrics" != "NOT_FOUND" ]]; then
                    IFS='|' read -r f1 em sql_em epoch best_ep <<< "$metrics"
                    update_results_table "$cfg_name" "$f1" "$em" "$sql_em" "${epoch} (best@${best_ep})" "—"
                    append_note "Config \`$cfg_name\` completed: F1=$f1, EM=$em, SQL_EM=$sql_em (best@epoch $best_ep / $epoch total)"
                fi
            else
                append_issue "Config $cfg_name FAILED in MLflow" \
                    "MLflow status = FAILED" \
                    "Training encountered a fatal error. Check train log for details." \
                    "Investigate stderr in train log. The sequential script will continue to the next config."
            fi
        fi
    done

    # 8) Log GPU status periodically
    gpu_info=$(check_gpu)
    if [[ -n "$gpu_info" ]]; then
        log "GPU: temp=$(echo $gpu_info | cut -d',' -f1)C, mem=$(echo $gpu_info | cut -d',' -f2)/$(echo $gpu_info | cut -d',' -f3)MiB, util=$(echo $gpu_info | cut -d',' -f4)%"
    fi

    sleep "$CHECK_INTERVAL"
done

log "Monitor script finished."
