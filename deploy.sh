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
PASSIVBOT_ARGS="python src/optimize.py strategies/hype_top_4pairs.json"
CONTAINER_NAME="passivbot_optimize"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine mode: deploy (default) or check
MODE="${1:-deploy}"

if [ "$MODE" == "deploy" ]; then
    echo -e "${GREEN}[1/3] Syncing local code to remote server...${NC}"
    # Using rsync to upload changes.
    rsync -avz -e "ssh -i \"$SSH_KEY_PATH\"" \
        --exclude '.git' \
        --exclude 'venv' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude 'backtests' \
        --exclude 'caches' \
        --exclude 'optimize_results' \
        ./ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

    echo -e "${GREEN}[2/3] Starting optimization in background...${NC}"
    ssh -i "$SSH_KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" << EOF
        set -e
        cd $REMOTE_DIR
        
        # Check and remove old container if exists
        if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
             docker rm -f ${CONTAINER_NAME} > /dev/null
        fi
        
        # Initialize results dirs if not exist to ensure permissions/existence
        mkdir -p optimize_results configs backtests
        
        echo "Starting container: ${CONTAINER_NAME}"
        # Start detached
        docker compose run -d --name ${CONTAINER_NAME} passivbot $PASSIVBOT_ARGS
EOF

    echo -e "${GREEN}[3/3] Optimization started. Logs will follow for 30 seconds.${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop watching logs (optimization will continue).${NC}"
    
    # Stream logs with timeout. The || true prevents script failure on meaningful exit codes from timeout/ssh
    ssh -t -i "$SSH_KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" "timeout 30s docker logs -f ${CONTAINER_NAME}" || true
    
    echo -e "\n${GREEN}Deployment complete. Optimization running in background.${NC}"
    echo "Use './deploy.sh check' to check status and retrieve results."

elif [ "$MODE" == "check" ]; then
    echo -e "${GREEN}Checking optimization status...${NC}"
    
    # Check if running
    IS_RUNNING=$(ssh -i "$SSH_KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$' && echo 'yes' || echo 'no'")
    
    if [ "$IS_RUNNING" == "yes" ]; then
        echo -e "${YELLOW}Optimization '${CONTAINER_NAME}' is still running.${NC}"
        echo "To view logs: ssh -i \"$SSH_KEY_PATH\" $REMOTE_USER@$REMOTE_HOST \"docker logs -f --tail 100 ${CONTAINER_NAME}\""
    else
        echo -e "${GREEN}Optimization finished. Downloading results...${NC}"
        
        ZIP_FILE="/tmp/passivbot_results_$$.zip"
        LINUX_TEMP_DIR="/tmp/passivbot_extract_$$"
        # Folders to retrieve
        FILES_TO_RETRIEVE=("configs" "backtests" "optimize_results")
        
        # Create zip on remote
        echo "Zipping remote files..."
        ssh -i "$SSH_KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR && zip -r /tmp/retrieved_results.zip ${FILES_TO_RETRIEVE[*]} -x \"*placeholder*\" || true"
        
        # SCP download
        echo "Downloading..."
        scp -i "$SSH_KEY_PATH" "$REMOTE_USER@$REMOTE_HOST:/tmp/retrieved_results.zip" "$ZIP_FILE"
        
        # Clean remote
        ssh -i "$SSH_KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" "rm -f /tmp/retrieved_results.zip"
        
        # Unzip locally
        echo "Extracting..."
        mkdir -p "$LINUX_TEMP_DIR"
        unzip -o "$ZIP_FILE" -d "$LINUX_TEMP_DIR"
        
        # Sync to workspace using robocopy logic from original script
        echo "Merging files into workspace..."
        
        for ITEM in "${FILES_TO_RETRIEVE[@]}"; do
            SRC="$LINUX_TEMP_DIR/$ITEM"
            DEST="$(pwd)/$ITEM"
            
            if [ -d "$SRC" ]; then
                # Convert to Windows path for robocopy (assumes WSL environment)
                WIN_SRC=$(wslpath -w "$SRC")
                WIN_DEST=$(wslpath -w "$DEST")
                echo "Syncing $ITEM..."
                cmd.exe /c "robocopy $WIN_SRC $WIN_DEST /E /NFL /NDL /NJH /NJS /NC /NS" || true
            fi
        done
        
        # Cleanup local
        rm -rf "$LINUX_TEMP_DIR"
        rm -f "$ZIP_FILE"
        
        echo -e "${GREEN}Done! Results updated in workspace.${NC}"
    fi

else
    echo "Usage: ./deploy.sh [deploy|check]"
    echo "  deploy : Sync code, start optimization detached, watch logs briefly. (Default)"
    echo "  check  : Check status. If finished, download results."
fi
