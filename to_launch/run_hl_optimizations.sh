#!/bin/bash
# Run 3 Hyperliquid HYPE optimizations sequentially
cd /home/ubuntu/passivbot
source venv/bin/activate

CONFIGS=(
    "to_launch/generated/config_obs_hyperliquid.json"
    "to_launch/generated/hype_top_hyperliquid.json"
    "to_launch/generated/rank1_hype_opt_hyperliquid.json"
)

TOTAL=${#CONFIGS[@]}
echo "=========================================="
echo "Starting $TOTAL Hyperliquid optimizations"
echo "Started at: $(date)"
echo "=========================================="

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    NUM=$((i + 1))
    BASENAME=$(basename "$CFG" .json)
    LOG="optimize_${BASENAME}.log"

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Starting: $BASENAME"
    echo "Config: $CFG"
    echo "Log: $LOG"
    echo "Time: $(date)"
    echo "=========================================="

    python src/optimize.py "$CFG" 2>&1 | tee "$LOG"

    EXIT_CODE=$?
    echo ""
    echo "[$NUM/$TOTAL] Finished: $BASENAME (exit code: $EXIT_CODE)"
    echo "Time: $(date)"
    echo ""
done

echo "=========================================="
echo "All $TOTAL optimizations completed!"
echo "Finished at: $(date)"
echo "=========================================="
