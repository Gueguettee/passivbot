#!/bin/bash

# --- Configuration ---
# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    # Use sed to strip Windows carriage returns before sourcing
    source <(sed 's/\r$//' "$SCRIPT_DIR/.env")
else
    echo "Error: .env file not found. Copy .env.example to .env and fill in your values."
    exit 1
fi

# Validate required variables
if [ -z "$REMOTE_USER" ] || [ -z "$REMOTE_HOST" ] || [ -z "$SSH_KEY_PATH" ]; then
    echo "Error: Missing required environment variables. Check your .env file."
    exit 1
fi

# Default values for optional variables
REMOTE_DIR="${REMOTE_DIR:-~/passivbot}"
COMMAND="docker compose run --rm passivbot python src/backtest.py strategies/hype_top_4pairs.json"

# --- Script ---
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}[1/3] Syncing local code to remote server...${NC}"
# Using rsync to upload change. Excludes heavy/unnecessary folders.
# Note: This requires Git Bash or WSL on Windows.
rsync -avz -e "ssh -i \"$SSH_KEY_PATH\"" \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'backtests' \
    --exclude 'caches' \
    ./ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

echo -e "${GREEN}[2/3] Executing command on remote server...${NC}"
ssh -i "$SSH_KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" << EOF
    set -e
    cd $REMOTE_DIR
    
    # Optional: Activate virtual environment if necessary
    # source venv/bin/activate
    
    echo "Running: $COMMAND"
    $COMMAND
EOF

echo -e "${GREEN}[3/3] Retrieving updated configs/results...${NC}"
# Use zip/unzip approach with Windows-native copy for WSL compatibility
ZIP_FILE="/tmp/passivbot_results_$$.zip"
LINUX_TEMP_DIR="/tmp/passivbot_extract_$$"
FILES_TO_RETRIEVE=("configs" "backtests")

# Create zip on remote server and download
ssh -i "$SSH_KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR && zip -r /tmp/retrieved_files.zip ${FILES_TO_RETRIEVE[*]}"
scp -i "$SSH_KEY_PATH" "$REMOTE_USER@$REMOTE_HOST:/tmp/retrieved_files.zip" "$ZIP_FILE"

# Extract to Linux temp dir (always works)
mkdir -p "$LINUX_TEMP_DIR"
unzip -o "$ZIP_FILE" -d "$LINUX_TEMP_DIR"

# Convert paths to Windows format
WIN_TEMP_CONFIGS=$(wslpath -w "$LINUX_TEMP_DIR/configs")
WIN_TEMP_BACKTESTS=$(wslpath -w "$LINUX_TEMP_DIR/backtests")
WIN_DEST_CONFIGS=$(wslpath -w "$(pwd)/configs")
WIN_DEST_BACKTESTS=$(wslpath -w "$(pwd)/backtests")

# Use robocopy (more reliable than xcopy) for each folder
cmd.exe /c "robocopy $WIN_TEMP_CONFIGS $WIN_DEST_CONFIGS /E /NFL /NDL /NJH /NJS /NC /NS" || true
cmd.exe /c "robocopy $WIN_TEMP_BACKTESTS $WIN_DEST_BACKTESTS /E /NFL /NDL /NJH /NJS /NC /NS" || true

rm -rf "$LINUX_TEMP_DIR"
rm -f "$ZIP_FILE"
# Cleanup remote temp file
ssh -i "$SSH_KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" "rm -f /tmp/retrieved_files.zip"

echo -e "${GREEN}Done!${NC}"
