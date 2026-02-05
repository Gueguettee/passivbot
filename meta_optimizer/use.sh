#!/bin/bash
#
# Meta-Optimizer Usage Guide
# Run this script to see all available commands and examples
#

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                     META-OPTIMIZER FOR PASSIVBOT                             ║
║                         Usage Guide                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF

echo -e "${BOLD}${CYAN}═══ WHAT IS IT? ═══${NC}"
echo ""
echo "The Meta-Optimizer finds ROBUST trading strategies that work across"
echo "multiple coins and time periods, avoiding overfitting."
echo ""
echo "It does this by:"
echo "  1. Splitting data into training/validation folds (time & coin based)"
echo "  2. Running Passivbot optimizer on each training fold"
echo "  3. Validating strategies on held-out data"
echo "  4. Scoring strategies on CONSISTENCY, not just raw performance"
echo ""

echo -e "${BOLD}${CYAN}═══ CONFIGURATION ═══${NC}"
echo ""
echo -e "${YELLOW}Step 1: Create .env file${NC}"
echo ""
echo "  cp meta_optimizer/.env.example meta_optimizer/.env"
echo ""
echo -e "${YELLOW}Step 2: Edit .env with your server details${NC}"
echo ""
echo "  REMOTE_HOST=ubuntu@your-server-ip"
echo "  SSH_KEY=~/.ssh/your-key.pem"
echo "  REMOTE_DIR=/home/ubuntu/passivbot"
echo ""

echo -e "${BOLD}${CYAN}═══ DEPLOYMENT COMMANDS ═══${NC}"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh setup${NC}"
echo "  First-time setup of remote instance"
echo "  - Installs Python, Rust, system dependencies"
echo "  - Creates directory structure"
echo "  - Only needed once per instance"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh deploy${NC}"
echo "  Sync code to remote instance"
echo "  - Uses rsync for efficient transfer"
echo "  - Sets up Python venv and installs requirements"
echo "  - Builds Rust extensions"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh test${NC}"
echo "  Run tests on remote instance"
echo "  - Verifies all modules import correctly"
echo "  - Tests splitting, scoring logic"
echo "  - Runs pytest if available"
echo ""

echo -e "${BOLD}${CYAN}═══ RUNNING OPTIMIZATION ═══${NC}"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh run --quick-test${NC}"
echo "  Start a quick test run (recommended first)"
echo "  - 2 time folds, 5000 iterations per fold"
echo "  - Takes ~1-2 hours"
echo "  - Good for verifying everything works"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh run${NC}"
echo "  Start full optimization with default settings"
echo "  - 3 time folds × 2 coin folds = 6 combined folds"
echo "  - 20000 iterations per fold"
echo "  - Takes ~8-24 hours depending on data"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh run --config path/to/config.json${NC}"
echo "  Start optimization with specific Passivbot config"
echo ""

echo -e "${BOLD}${CYAN}═══ MONITORING ═══${NC}"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh monitor${NC}"
echo "  Interactive monitoring on remote"
echo "  - Shows CPU/memory usage"
echo "  - Shows optimization progress"
echo "  - Shows recent logs"
echo ""

echo -e "${GREEN}python meta_optimizer/monitor_dashboard.py${NC}"
echo "  Visual dashboard (runs locally)"
echo "  - Real-time progress bars"
echo "  - Fold completion status"
echo "  - Best robustness score found"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh logs${NC}"
echo "  Tail logs from remote instance"
echo "  - Shows latest optimization output"
echo "  - Use --lines N to show more lines"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh status${NC}"
echo "  Check instance status"
echo "  - System resources (CPU, memory, disk)"
echo "  - Running processes"
echo "  - Latest results directories"
echo ""

echo -e "${GREEN}ssh \$REMOTE_HOST -t 'tmux attach -t meta_opt'${NC}"
echo "  Attach directly to tmux session"
echo "  - See live output"
echo "  - Detach with Ctrl+B, then D"
echo ""

echo -e "${BOLD}${CYAN}═══ MANAGING RUNS ═══${NC}"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh stop${NC}"
echo "  Stop running optimization"
echo "  - Kills tmux session"
echo "  - Progress is saved (can resume later)"
echo ""

echo -e "${GREEN}./meta_optimizer/deploy.sh download --output ./results${NC}"
echo "  Download results to local machine"
echo "  - Gets all results and logs"
echo "  - Best configs in best_configs/ folder"
echo ""

echo -e "${BOLD}${CYAN}═══ FULL WORKFLOW EXAMPLE ═══${NC}"
echo ""
echo -e "${YELLOW}# First time setup${NC}"
echo "cp meta_optimizer/.env.example meta_optimizer/.env"
echo "# Edit .env with your server details"
echo "./meta_optimizer/deploy.sh setup"
echo ""
echo -e "${YELLOW}# Deploy and test${NC}"
echo "./meta_optimizer/deploy.sh deploy"
echo "./meta_optimizer/deploy.sh test"
echo ""
echo -e "${YELLOW}# Run quick test first${NC}"
echo "./meta_optimizer/deploy.sh run --quick-test"
echo ""
echo -e "${YELLOW}# Monitor in another terminal${NC}"
echo "python meta_optimizer/monitor_dashboard.py"
echo ""
echo -e "${YELLOW}# When done, download results${NC}"
echo "./meta_optimizer/deploy.sh download --output ./my_results"
echo ""
echo -e "${YELLOW}# Check best configs${NC}"
echo "ls ./my_results/*/best_configs/"
echo "cat ./my_results/*/best_configs/rank_1.json"
echo ""

echo -e "${BOLD}${CYAN}═══ OUTPUT STRUCTURE ═══${NC}"
echo ""
echo "meta_optimize_results/YYYY-MM-DD_HHMMSS/"
echo "├── meta_config.json        # Configuration used"
echo "├── checkpoint.json         # Progress (for resume)"
echo "├── summary.json            # Final results summary"
echo "├── all_ranked_configs.json # All configs with scores"
echo "├── fold_results/"
echo "│   ├── fold_0/"
echo "│   │   ├── training/       # Optimizer output"
echo "│   │   └── validation/     # Validation backtests"
echo "│   └── fold_1/"
echo "└── best_configs/"
echo "    ├── rank_1.json         # Most robust config"
echo "    ├── rank_2.json"
echo "    └── ..."
echo ""

echo -e "${BOLD}${CYAN}═══ ROBUSTNESS SCORING ═══${NC}"
echo ""
echo "Strategies are scored on:"
echo ""
echo "  ${BOLD}Consistency (40%)${NC}"
echo "    - Low variance of performance across folds"
echo "    - High percentage of profitable folds"
echo ""
echo "  ${BOLD}Degradation (30%)${NC}"
echo "    - Ratio of validation to training performance"
echo "    - Detects overfitting (train >> validation = bad)"
echo ""
echo "  ${BOLD}Worst Case (20%)${NC}"
echo "    - Performance in the worst fold"
echo "    - Ensures no catastrophic failures"
echo ""
echo "  ${BOLD}Stability (10%)${NC}"
echo "    - Parameter sensitivity"
echo "    - Robust to small changes"
echo ""

echo -e "${BOLD}${CYAN}═══ TIPS ═══${NC}"
echo ""
echo "• Always run --quick-test first to verify setup"
echo "• Use tmux attach to see detailed progress"
echo "• Check logs if something fails: ./meta_optimizer/deploy.sh logs"
echo "• Results are checkpointed - you can resume after interruption"
echo "• Top configs are in best_configs/rank_1.json, rank_2.json, etc."
echo ""

echo -e "${BOLD}${CYAN}═══ NEED HELP? ═══${NC}"
echo ""
echo "./meta_optimizer/deploy.sh --help    # Show deploy commands"
echo "python meta_optimizer/cli.py --help  # Show CLI options"
echo ""
