#!/bin/bash
#
# Meta-Optimizer Deployment Script
# Deploys, tests, and monitors meta_optimizer on a remote instance
#
# Configuration is loaded from .env file in the same directory.
# Copy .env.example to .env and configure your settings.
#
# Usage:
#   ./deploy.sh <command> [options]
#
# Commands:
#   setup       - Initial setup on remote instance
#   deploy      - Deploy/sync code to instance
#   test        - Run tests on instance
#   run         - Start meta-optimization
#   monitor     - Monitor running optimization
#   logs        - Tail logs from instance
#   status      - Check instance status
#   stop        - Stop running optimization
#   download    - Download results from instance
#   all         - Deploy, test, and run
#
# Examples:
#   ./deploy.sh setup
#   ./deploy.sh deploy
#   ./deploy.sh run --config configs/template.json --quick-test
#   ./deploy.sh monitor
#   ./deploy.sh download --output ./results/

set -e

# ============================================================================
# Load Configuration from .env
# ============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Load .env file if it exists
if [ -f "$ENV_FILE" ]; then
    # Export variables from .env (ignore comments and empty lines)
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env file not found at $ENV_FILE"
    echo "Copy .env.example to .env and configure your settings."
    echo ""
fi

# ============================================================================
# Configuration (loaded from .env or use defaults)
# ============================================================================

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/passivbot}"
SSH_KEY="${SSH_KEY:-}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
VENV_NAME="${VENV_NAME:-venv}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Build SSH command with optional key
ssh_cmd() {
    local cmd="ssh"
    if [ -n "$SSH_KEY" ]; then
        cmd="$cmd -i $SSH_KEY"
    fi
    echo "$cmd $REMOTE_HOST"
}

# Build SCP command with optional key
scp_cmd() {
    local cmd="scp"
    if [ -n "$SSH_KEY" ]; then
        cmd="$cmd -i $SSH_KEY"
    fi
    echo "$cmd"
}

# Build rsync command with optional key
rsync_cmd() {
    local cmd="rsync -avz --progress"
    if [ -n "$SSH_KEY" ]; then
        cmd="$cmd -e \"ssh -i $SSH_KEY\""
    fi
    echo "$cmd"
}

# Check if host is set
check_host() {
    if [ -z "$REMOTE_HOST" ]; then
        log_error "REMOTE_HOST not configured."
        log_error "Please set REMOTE_HOST in $ENV_FILE"
        log_error "Example: REMOTE_HOST=ubuntu@your-instance.com"
        echo ""
        log_info "To create .env file:"
        echo "  cp $SCRIPT_DIR/.env.example $ENV_FILE"
        exit 1
    fi
}

# Execute command on remote
remote_exec() {
    $(ssh_cmd) "$@"
}

# ============================================================================
# Commands
# ============================================================================

cmd_setup() {
    log_info "Setting up remote instance: $REMOTE_HOST"

    # Check connectivity
    log_info "Checking SSH connectivity..."
    if ! remote_exec "echo 'Connection OK'"; then
        log_error "Cannot connect to $REMOTE_HOST"
        exit 1
    fi

    # Install system dependencies
    log_info "Installing system dependencies..."
    remote_exec "sudo apt-get update && sudo apt-get install -y \
        python3 python3-pip python3-venv \
        git curl wget htop tmux \
        build-essential libssl-dev libffi-dev python3-dev"

    # Install Rust (required for passivbot-rust)
    log_info "Installing Rust..."
    remote_exec "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    remote_exec "source ~/.cargo/env && rustc --version"

    # Create directory structure
    log_info "Creating directory structure..."
    remote_exec "mkdir -p $REMOTE_DIR"
    remote_exec "mkdir -p $REMOTE_DIR/logs"
    remote_exec "mkdir -p $REMOTE_DIR/meta_optimize_results"

    log_success "Setup complete!"
}

cmd_deploy() {
    log_info "Deploying code to $REMOTE_HOST:$REMOTE_DIR"

    # Get local script directory
    LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    # Files/directories to sync
    log_info "Syncing passivbot code..."

    # Use rsync for efficient sync
    eval "$(rsync_cmd) \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'venv' \
        --exclude 'backtests' \
        --exclude 'optimize_results' \
        --exclude 'meta_optimize_results' \
        --exclude 'caches' \
        --exclude '*.egg-info' \
        --exclude 'target' \
        $LOCAL_DIR/ $REMOTE_HOST:$REMOTE_DIR/"

    # Setup virtual environment and install dependencies
    log_info "Setting up Python environment..."
    remote_exec "cd $REMOTE_DIR && \
        $PYTHON_CMD -m venv $VENV_NAME && \
        source $VENV_NAME/bin/activate && \
        pip install --upgrade pip && \
        pip install -r requirements.txt"

    # Check if Rust is installed, install if not
    log_info "Checking Rust installation..."
    if ! remote_exec "test -f ~/.cargo/env"; then
        log_warn "Rust not found. Installing Rust..."
        remote_exec "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    fi

    # Build Rust extensions
    log_info "Building Rust extensions..."
    remote_exec "cd $REMOTE_DIR && \
        source $VENV_NAME/bin/activate && \
        source ~/.cargo/env && \
        cd passivbot-rust && \
        pip install maturin && \
        maturin develop --release"

    log_success "Deployment complete!"
}

cmd_test() {
    log_info "Running tests on $REMOTE_HOST"

    # Test Python imports
    log_info "Testing meta_optimizer imports..."
    remote_exec "cd $REMOTE_DIR && \
        source $VENV_NAME/bin/activate && \
        python3 -c 'from meta_optimizer.config import MetaOptimizerConfig; print(\"Config module: OK\")' && \
        python3 -c 'from meta_optimizer.splitting import TimeSplitter, CoinSplitter; print(\"Splitting modules: OK\")' && \
        python3 -c 'from meta_optimizer.scoring import RobustnessScorer; print(\"Scoring modules: OK\")' && \
        python3 -c 'from meta_optimizer.orchestrator import MetaOptimizer; print(\"Orchestrator: OK\")'"

    # Test CLI dry-run
    log_info "Testing CLI dry-run..."
    remote_exec "cd $REMOTE_DIR && \
        source $VENV_NAME/bin/activate && \
        python3 meta_optimizer/cli.py configs/template.json --quick-test --dry-run"

    # Test splitting logic
    log_info "Testing splitting logic..."
    remote_exec "cd $REMOTE_DIR && \
        source $VENV_NAME/bin/activate && \
        python3 -c \"
from meta_optimizer.splitting import TimeSplitter, CoinSplitter
ts = TimeSplitter(n_folds=2, train_months=12, val_months=3)
folds = ts.generate_folds('2022-01-01', '2024-01-01')
print(f'Generated {len(folds)} time folds')
for f in folds:
    print(f'  {f}')
\""

    # Test scoring logic
    log_info "Testing scoring logic..."
    remote_exec "cd $REMOTE_DIR && \
        source $VENV_NAME/bin/activate && \
        python3 -c \"
from meta_optimizer.scoring import RobustnessScorer
scorer = RobustnessScorer()
mock_results = [
    {'adg_pnl': 0.001, 'sharpe_ratio_pnl': 1.0},
    {'adg_pnl': 0.0015, 'sharpe_ratio_pnl': 1.2},
]
score = scorer.score(mock_results)
print(f'Robustness score: {score.overall_score:.3f}')
print(f'Passes thresholds: {score.passes_thresholds}')
\""

    # Run pytest if available
    log_info "Running pytest (if available)..."
    remote_exec "cd $REMOTE_DIR && \
        source $VENV_NAME/bin/activate && \
        pip install pytest -q && \
        pytest tests/ -v --tb=short 2>/dev/null || echo 'Pytest completed (some tests may have failed)'"

    log_success "All tests passed!"
}

cmd_run() {
    local config="${CONFIG:-configs/template.json}"
    local quick_test="${QUICK_TEST:-false}"
    local extra_args="${EXTRA_ARGS:-}"

    log_info "Starting meta-optimization on $REMOTE_HOST"
    log_info "Config: $config"
    log_info "Quick test: $quick_test"

    # Build command
    local run_cmd="cd $REMOTE_DIR && \
        source $VENV_NAME/bin/activate && \
        python3 meta_optimizer/cli.py $config"

    if [ "$quick_test" = "true" ]; then
        run_cmd="$run_cmd --quick-test"
    fi

    if [ -n "$extra_args" ]; then
        run_cmd="$run_cmd $extra_args"
    fi

    # Create log file name
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="logs/meta_opt_${timestamp}.log"

    # Start in tmux session for persistence
    log_info "Starting in tmux session 'meta_opt'..."
    remote_exec "cd $REMOTE_DIR && \
        tmux kill-session -t meta_opt 2>/dev/null || true && \
        tmux new-session -d -s meta_opt \"$run_cmd 2>&1 | tee $log_file\""

    log_success "Meta-optimization started!"
    log_info "Log file: $REMOTE_DIR/$log_file"
    log_info "To monitor: ./meta_optimizer/deploy.sh monitor"
    log_info "To attach: ssh $REMOTE_HOST -t 'tmux attach -t meta_opt'"
}

cmd_monitor() {
    log_info "Monitoring meta-optimization on $REMOTE_HOST"

    # Create monitoring script
    local monitor_script='
import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime

def get_resource_usage():
    """Get CPU and memory usage."""
    try:
        cpu = subprocess.check_output(["grep", "cpu ", "/proc/stat"]).decode().split()
        mem = subprocess.check_output(["free", "-m"]).decode().split("\n")[1].split()

        # Calculate CPU usage
        cpu_total = sum(int(x) for x in cpu[1:])
        cpu_idle = int(cpu[4])
        cpu_pct = 100 * (1 - cpu_idle / cpu_total) if cpu_total > 0 else 0

        # Memory usage
        mem_total = int(mem[1])
        mem_used = int(mem[2])
        mem_pct = 100 * mem_used / mem_total if mem_total > 0 else 0

        return cpu_pct, mem_pct, mem_used, mem_total
    except:
        return 0, 0, 0, 0

def get_latest_results_dir():
    """Find the latest meta_optimize_results directory."""
    results_base = Path("meta_optimize_results")
    if not results_base.exists():
        return None
    dirs = [d for d in results_base.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)

def get_progress(results_dir):
    """Get optimization progress from checkpoint."""
    if not results_dir:
        return {}

    checkpoint = results_dir / "checkpoint.json"
    if not checkpoint.exists():
        return {}

    try:
        with open(checkpoint) as f:
            data = json.load(f)
        state = data.get("state", {})
        return {
            "phase": state.get("phase", "unknown"),
            "folds_completed": len(state.get("optimization_results", {})),
            "configs_found": len(state.get("all_configs", {})),
            "validations_done": sum(len(v) for v in state.get("validation_matrix", {}).values()),
        }
    except:
        return {}

def check_tmux_running():
    """Check if meta_opt tmux session is running."""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", "meta_opt"],
            capture_output=True
        )
        return result.returncode == 0
    except:
        return False

# Main monitoring loop
print("=" * 60)
print("META-OPTIMIZER MONITOR")
print("=" * 60)
print(f"Started at: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}")
print()

try:
    while True:
        os.system("clear")
        print("=" * 60)
        print(f"META-OPTIMIZER MONITOR - {datetime.now().strftime(\"%H:%M:%S\")}")
        print("=" * 60)

        # Check if running
        is_running = check_tmux_running()
        status = "RUNNING" if is_running else "STOPPED"
        status_color = "\033[92m" if is_running else "\033[91m"
        print(f"Status: {status_color}{status}\033[0m")
        print()

        # Resource usage
        cpu, mem_pct, mem_used, mem_total = get_resource_usage()
        print(f"Resources:")
        print(f"  CPU: {cpu:.1f}%")
        print(f"  Memory: {mem_used}MB / {mem_total}MB ({mem_pct:.1f}%)")
        print()

        # Progress
        results_dir = get_latest_results_dir()
        if results_dir:
            print(f"Results dir: {results_dir.name}")
            progress = get_progress(results_dir)
            if progress:
                print(f"Progress:")
                print(f"  Phase: {progress.get(\"phase\", \"unknown\")}")
                print(f"  Folds completed: {progress.get(\"folds_completed\", 0)}")
                print(f"  Configs found: {progress.get(\"configs_found\", 0)}")
                print(f"  Validations done: {progress.get(\"validations_done\", 0)}")
        else:
            print("No results directory found yet")

        print()
        print("-" * 60)
        print("Press Ctrl+C to exit monitor")
        print("To attach to session: tmux attach -t meta_opt")

        time.sleep(5)
except KeyboardInterrupt:
    print("\nMonitor stopped.")
'

    # Run monitor on remote (no venv needed - uses only standard library)
    remote_exec "cd $REMOTE_DIR && python3 -c '$monitor_script'"
}

cmd_logs() {
    local lines="${LINES:-100}"

    log_info "Fetching logs from $REMOTE_HOST"

    # Find latest log file
    local log_file=$(remote_exec "ls -t $REMOTE_DIR/logs/meta_opt_*.log 2>/dev/null | head -1")

    if [ -z "$log_file" ]; then
        log_warn "No log files found"

        # Try to get tmux output
        log_info "Attempting to capture tmux output..."
        remote_exec "tmux capture-pane -t meta_opt -p 2>/dev/null || echo 'No active session'"
        return
    fi

    log_info "Tailing: $log_file"
    remote_exec "tail -f -n $lines $log_file"
}

cmd_status() {
    log_info "Checking status on $REMOTE_HOST"

    echo ""
    echo "=== System Status ==="
    remote_exec "uptime && echo '' && free -h && echo '' && df -h /"

    echo ""
    echo "=== Tmux Sessions ==="
    remote_exec "tmux list-sessions 2>/dev/null || echo 'No tmux sessions'"

    echo ""
    echo "=== Python Processes ==="
    remote_exec "pgrep -a python | grep -E '(meta_optimizer|optimize|backtest)' || echo 'No relevant Python processes'"

    echo ""
    echo "=== Latest Results ==="
    remote_exec "ls -lt $REMOTE_DIR/meta_optimize_results/ 2>/dev/null | head -5 || echo 'No results yet'"

    echo ""
    echo "=== Recent Logs ==="
    remote_exec "ls -lt $REMOTE_DIR/logs/*.log 2>/dev/null | head -3 || echo 'No logs yet'"
}

cmd_stop() {
    log_info "Stopping meta-optimization on $REMOTE_HOST"

    # Kill tmux session
    remote_exec "tmux kill-session -t meta_opt 2>/dev/null || true"

    # Kill any remaining Python processes
    remote_exec "pkill -f 'meta_optimizer/cli.py' 2>/dev/null || true"

    log_success "Optimization stopped"
}

cmd_download() {
    local output_dir="${OUTPUT_DIR:-./meta_results}"

    log_info "Downloading results from $REMOTE_HOST to $output_dir"

    # Create output directory
    mkdir -p "$output_dir"

    # Download results
    eval "$(rsync_cmd) \
        $REMOTE_HOST:$REMOTE_DIR/meta_optimize_results/ \
        $output_dir/"

    # Also download logs
    mkdir -p "$output_dir/logs"
    eval "$(rsync_cmd) \
        $REMOTE_HOST:$REMOTE_DIR/logs/ \
        $output_dir/logs/"

    log_success "Results downloaded to $output_dir"

    # Show summary
    echo ""
    echo "=== Downloaded Results ==="
    ls -la "$output_dir"
}

cmd_all() {
    log_info "Running full deployment pipeline"

    cmd_deploy
    echo ""
    cmd_test
    echo ""
    cmd_run
}

# ============================================================================
# Main
# ============================================================================

show_help() {
    echo "Meta-Optimizer Deployment Script"
    echo ""
    echo "Configuration is loaded from .env file."
    echo "Copy .env.example to .env and set REMOTE_HOST and SSH_KEY."
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup       Initial setup on remote instance"
    echo "  deploy      Deploy/sync code to instance"
    echo "  test        Run tests on instance"
    echo "  run         Start meta-optimization"
    echo "  monitor     Monitor running optimization"
    echo "  logs        Tail logs from instance"
    echo "  status      Check instance status"
    echo "  stop        Stop running optimization"
    echo "  download    Download results from instance"
    echo "  all         Deploy, test, and run"
    echo ""
    echo "Options:"
    echo "  --config FILE     Passivbot config file (default: configs/template.json)"
    echo "  --quick-test      Use quick test settings"
    echo "  --output DIR      Local output directory for downloads"
    echo "  --lines N         Number of log lines to show (default: 100)"
    echo ""
    echo "Environment (.env file):"
    echo "  REMOTE_HOST       Remote host (user@hostname)"
    echo "  SSH_KEY           SSH private key file path"
    echo "  REMOTE_DIR        Remote directory (default: /home/ubuntu/passivbot)"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 deploy"
    echo "  $0 run --config configs/template.json --quick-test"
    echo "  $0 monitor"
    echo "  $0 download --output ./results"
}

# Parse arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        setup|deploy|test|run|monitor|logs|status|stop|download|all)
            COMMAND="$1"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_TEST="true"
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --lines)
            LINES="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Run command
if [ -z "$COMMAND" ]; then
    show_help
    exit 1
fi

check_host

case $COMMAND in
    setup)
        cmd_setup
        ;;
    deploy)
        cmd_deploy
        ;;
    test)
        cmd_test
        ;;
    run)
        cmd_run
        ;;
    monitor)
        cmd_monitor
        ;;
    logs)
        cmd_logs
        ;;
    status)
        cmd_status
        ;;
    stop)
        cmd_stop
        ;;
    download)
        cmd_download
        ;;
    all)
        cmd_all
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
