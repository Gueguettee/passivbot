#!/bin/bash
# Run all 9 HYPE optimizations sequentially (3 configs × 3 exchanges)
# Each uses 8 CPUs and 200K iterations

cd /home/ubuntu/passivbot
source venv/bin/activate

CONFIGS=(
    "to_launch/generated/config_obs_bybit.json"
    "to_launch/generated/config_obs_binance.json"
    "to_launch/generated/config_obs_hyperliquid.json"
    "to_launch/generated/hype_top_bybit.json"
    "to_launch/generated/hype_top_binance.json"
    "to_launch/generated/hype_top_hyperliquid.json"
    "to_launch/generated/rank1_hype_opt_bybit.json"
    "to_launch/generated/rank1_hype_opt_binance.json"
    "to_launch/generated/rank1_hype_opt_hyperliquid.json"
)

TOTAL=${#CONFIGS[@]}
echo "=========================================="
echo "Starting $TOTAL optimizations sequentially"
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
