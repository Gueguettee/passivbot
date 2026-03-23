#!/bin/bash
# Passivbot Live Deployment Script
# Inspired by cBot-Project-Gueguette/deploy.sh
#
# Usage:
#   ./deploy.sh deploy      Sync code, build Docker image, start live bot
#   ./deploy.sh stop        Stop the live bot container
#   ./deploy.sh restart     Stop then redeploy
#   ./deploy.sh logs [-f]   View logs (use -f to follow, -n NUM for line count)
#   ./deploy.sh status      Check container status and resource usage
#   ./deploy.sh connect     SSH into the remote server
#
# Add --hl to any command to target Hyperliquid instead of Binance:
#   ./deploy.sh deploy --hl
#   ./deploy.sh logs -f --hl

set -e

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source <(sed 's/\r$//' "$SCRIPT_DIR/.env")
else
    echo "Error: .env file not found. Copy .env.example to .env and fill in your values."
    exit 1
fi

# Validate required variables
if [ -z "$REMOTE_USER" ] || [ -z "$REMOTE_HOST" ] || [ -z "$SSH_KEY_PATH" ]; then
    echo "Error: Missing required environment variables (REMOTE_USER, REMOTE_HOST, SSH_KEY_PATH)."
    echo "Check your .env file."
    exit 1
fi

REMOTE_DIR="${REMOTE_DIR:-~/passivbot}"
SSH_OPTS="-o ServerAliveInterval=60 -o ServerAliveCountMax=120"

# Exchange selection: pass --hl anywhere to target Hyperliquid
EXCHANGE="binance"
for arg in "$@"; do
    if [ "$arg" == "--hl" ]; then
        EXCHANGE="hyperliquid"
    fi
done

if [ "$EXCHANGE" == "hyperliquid" ]; then
    CONTAINER_NAME="passivbot-live-hl"
    LIVE_CONFIG="configs/live_rank1_hype_hl.json"
    DOCKER_PROFILE="live-hl"
else
    CONTAINER_NAME="passivbot-live"
    LIVE_CONFIG="configs/live_rank1_hype.json"
    DOCKER_PROFILE="live"
fi

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Resolve SSH key path (handle Windows Git Bash / WSL)
resolve_ssh_key() {
    local key="$SSH_KEY_PATH"
    # Expand $HOME / ~ if present
    key="${key/#\~/$HOME}"
    # If on Windows Git Bash and path starts with /c/, try Windows path
    if [ ! -f "$key" ] && [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
        local win_key="$USERPROFILE/.ssh/$(basename "$key")"
        win_key="${win_key//\\//}"
        if [ -f "$win_key" ]; then
            key="$win_key"
        fi
    fi
    echo "$key"
}

SSH_KEY="$(resolve_ssh_key)"
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}Error: SSH key not found at $SSH_KEY${NC}"
    exit 1
fi

ssh_cmd() {
    ssh -i "$SSH_KEY" $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "$@"
}

# --- Commands ---

sync_to_remote() {
    echo -e "${GREEN}[sync] Syncing code to $REMOTE_HOST...${NC}"
    rsync -avz --delete \
        -e "ssh -i \"$SSH_KEY\"" \
        --exclude '.git' \
        --exclude 'venv' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude 'caches' \
        --exclude 'optimize_results' \
        --exclude 'backtests' \
        --exclude 'backtests_*' \
        --exclude 'optimization_results_*' \
        --exclude 'meta_optimizer' \
        --exclude 'to_launch' \
        --exclude '.env' \
        --exclude '*.log' \
        --exclude 'node_modules' \
        "$SCRIPT_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
    echo -e "${GREEN}[sync] Done.${NC}"
}

do_deploy() {
    sync_to_remote

    echo -e "${GREEN}[build] Building Docker image on server...${NC}"
    ssh_cmd -t "
        cd \"$REMOTE_DIR\"
        docker compose --profile $DOCKER_PROFILE build
    "

    echo -e "${GREEN}[start] Starting live bot...${NC}"
    ssh_cmd "
        cd \"$REMOTE_DIR\"
        # Stop existing container if running
        docker compose --profile $DOCKER_PROFILE down --remove-orphans 2>/dev/null || true
        docker compose --profile $DOCKER_PROFILE up -d
    "

    echo -e "${GREEN}[logs] Showing initial logs (30s)...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop watching (bot continues running).${NC}"
    ssh_cmd -t "timeout 30s docker logs -f $CONTAINER_NAME 2>&1" || true

    echo ""
    echo -e "${GREEN}Deployment complete. Bot running in background.${NC}"
    echo "  ./deploy.sh logs -f    Follow logs"
    echo "  ./deploy.sh status     Check status"
    echo "  ./deploy.sh stop       Stop bot"
}

do_stop() {
    echo -e "${YELLOW}[stop] Stopping live bot...${NC}"
    ssh_cmd "
        cd \"$REMOTE_DIR\"
        docker compose --profile $DOCKER_PROFILE down --remove-orphans
    "
    echo -e "${GREEN}Bot stopped.${NC}"
}

do_restart() {
    do_stop
    do_deploy
}

do_logs() {
    shift  # remove 'logs' from args
    local follow=""
    local lines="100"
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--follow) follow="-f"; shift ;;
            -n|--lines) lines="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    echo -e "${GREEN}[logs] Fetching logs from $REMOTE_HOST...${NC}"
    ssh_cmd -t "docker logs --tail=$lines $follow $CONTAINER_NAME 2>&1" || true
}

do_status() {
    echo -e "${GREEN}[status] Checking live bot on $REMOTE_HOST...${NC}"
    ssh_cmd "
        cd \"$REMOTE_DIR\"
        echo ''
        echo '=== Docker Containers ==='
        docker compose --profile $DOCKER_PROFILE ps
        echo ''
        echo '=== Resource Usage ==='
        docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' 2>/dev/null || echo '(no running containers)'
        echo ''
        echo '=== Config ==='
        echo 'Config: $LIVE_CONFIG'
        echo ''
        echo '=== Disk Usage ==='
        df -h / | tail -1
    "
}

do_connect() {
    echo -e "${GREEN}[connect] Connecting to $REMOTE_HOST...${NC}"
    ssh -i "$SSH_KEY" -t "$REMOTE_USER@$REMOTE_HOST" "cd \"$REMOTE_DIR\" && exec \$SHELL -l"
}

# --- Main ---
MODE="${1:-help}"

case "$MODE" in
    deploy)
        do_deploy
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_restart
        ;;
    logs)
        do_logs "$@"
        ;;
    status)
        do_status
        ;;
    connect)
        do_connect
        ;;
    *)
        echo "Passivbot Live Deployment"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  deploy       Sync code, build Docker image, start live bot"
        echo "  stop         Stop the live bot container"
        echo "  restart      Stop then redeploy"
        echo "  logs [-f]    View logs (use -f to follow, -n NUM for line count)"
        echo "  status       Check container status and resource usage"
        echo "  connect      SSH into the remote server"
        echo ""
        echo "Options:"
        echo "  --hl         Target Hyperliquid instead of Binance"
        ;;
esac
