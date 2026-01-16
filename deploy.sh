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
COMMAND="docker compose run --rm passivbot python src/optimize.py strategies/hype_top_4pairs.json"

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
# Downloading back specific folders where results might be saved
# Adjust this to match where your optimization saves data (e.g. configs/ or results/)
rsync -avz -e "ssh -i \"$SSH_KEY_PATH\"" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/configs/" ./configs/

echo -e "${GREEN}Done!${NC}"
