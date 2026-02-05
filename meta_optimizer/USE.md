# Meta-Optimizer for Passivbot

A robust strategy discovery system that finds trading strategies which work across multiple coins and time periods, avoiding overfitting.

## Quick Start

```bash
# 1. Configure your server
cp meta_optimizer/.env.example meta_optimizer/.env
# Edit .env with your server details

# 2. Deploy to server
./meta_optimizer/deploy.sh deploy

# 3. Run optimization
./meta_optimizer/deploy.sh run --quick-test

# 4. Monitor progress
./meta_optimizer/deploy.sh monitor
```

---

## Configuration

### `.env` File

Create `meta_optimizer/.env` with your server details:

```env
REMOTE_HOST=ubuntu@your-server-ip
SSH_KEY=~/.ssh/your-key.pem
REMOTE_DIR=/home/ubuntu/passivbot
```

| Variable | Description | Default |
|----------|-------------|---------|
| `REMOTE_HOST` | SSH connection string (user@host) | *required* |
| `SSH_KEY` | Path to SSH private key | *optional* |
| `REMOTE_DIR` | Directory on remote server | `/home/ubuntu/passivbot` |
| `PYTHON_CMD` | Python command | `python3` |
| `VENV_NAME` | Virtual environment name | `venv` |

---

## Commands

### Setup & Deployment

#### `setup` - First-time instance setup

```bash
./meta_optimizer/deploy.sh setup
```

Installs on remote server:
- Python 3, pip, venv
- Rust (for passivbot-rust)
- System dependencies (build-essential, etc.)
- Creates directory structure

**Run once per new instance.**

---

#### `deploy` - Sync code to server

```bash
./meta_optimizer/deploy.sh deploy
```

- Syncs passivbot code via rsync
- Creates Python virtual environment
- Installs Python dependencies
- Builds Rust extensions

**Run after code changes.**

---

#### `test` - Verify installation

```bash
./meta_optimizer/deploy.sh test
```

- Tests all module imports
- Runs CLI dry-run
- Runs pytest (if available)

---

### Running Optimization

#### `run` - Start meta-optimization

```bash
# Quick test (recommended first)
./meta_optimizer/deploy.sh run --quick-test

# Full optimization
./meta_optimizer/deploy.sh run

# With specific config
./meta_optimizer/deploy.sh run --config configs/my_config.json
```

| Option | Description |
|--------|-------------|
| `--quick-test` | Fast mode: 2 folds, 5k iterations (~1-2 hours) |
| `--config FILE` | Use specific Passivbot config file |

**Runs in tmux session for persistence.**

---

#### `all` - Deploy + Test + Run

```bash
./meta_optimizer/deploy.sh all --quick-test
```

Convenience command that runs deploy, test, and run in sequence.

---

### Monitoring

#### `monitor` - Interactive monitoring

```bash
./meta_optimizer/deploy.sh monitor
```

Shows on remote server:
- CPU/Memory usage
- Process status (running/stopped)
- Optimization progress
- Recent logs

**Press Ctrl+C to exit.**

---

#### `logs` - View logs

```bash
# Default (100 lines)
./meta_optimizer/deploy.sh logs

# More lines
./meta_optimizer/deploy.sh logs --lines 500
```

Tails the latest log file.

---

#### `status` - Check instance status

```bash
./meta_optimizer/deploy.sh status
```

Shows:
- System resources (uptime, memory, disk)
- Running tmux sessions
- Python processes
- Latest results directories

---

#### Python Dashboard (Local)

```bash
python meta_optimizer/monitor_dashboard.py
```

Visual dashboard running locally with:
- Progress bars
- Real-time updates
- Color-coded status

---

#### Attach to Session

```bash
ssh ubuntu@your-server -t 'tmux attach -t meta_opt'
```

See live output. Detach with `Ctrl+B`, then `D`.

---

### Managing Runs

#### `stop` - Stop optimization

```bash
./meta_optimizer/deploy.sh stop
```

Stops the running optimization. Progress is saved and can be resumed.

---

#### `download` - Get results

```bash
./meta_optimizer/deploy.sh download --output ./my_results
```

Downloads:
- All result directories
- Log files
- Best configurations

---

## Output Structure

```
meta_optimize_results/YYYY-MM-DD_HHMMSS/
├── meta_config.json          # Configuration used
├── checkpoint.json           # Progress state (for resume)
├── summary.json              # Final results
├── all_ranked_configs.json   # All configs with scores
├── fold_results/
│   ├── fold_0/
│   │   ├── training/         # Optimizer output
│   │   └── validation/       # Backtest results
│   └── fold_1/
└── best_configs/
    ├── rank_1.json           # Most robust config
    ├── rank_2.json
    └── ...
```

---

## Robustness Scoring

Strategies are scored on 4 components:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Consistency** | 40% | Low variance across folds |
| **Degradation** | 30% | Train vs validation ratio (detects overfitting) |
| **Worst Case** | 20% | Performance in worst fold |
| **Stability** | 10% | Parameter sensitivity |

### Thresholds

Default thresholds for a "passing" strategy:

| Threshold | Default | Description |
|-----------|---------|-------------|
| `min_profitable_folds_pct` | 80% | At least 80% of folds profitable |
| `max_degradation_ratio` | 0.5 | Val performance >= 50% of train |
| `max_cv_adg` | 1.0 | Coefficient of variation < 100% |
| `min_worst_fold_adg` | 0.0 | Worst fold at least break-even |

---

## Validation Schemes

### Time-Based (Walk-Forward)

```
Fold 1: Train 2021-2022, Validate 2023
Fold 2: Train 2022-2023, Validate 2024
Fold 3: Train 2023-2024, Validate 2025
```

### Coin-Based (K-Fold)

```
Fold 1: Train [BTC,ETH,SOL...], Validate [ADA,XRP,DOGE...]
Fold 2: Train [ADA,XRP,DOGE...], Validate [BTC,ETH,SOL...]
```

### Combined (Most Rigorous)

Both time AND coin holdout simultaneously.

---

## Configuration Options

### Quick Test Settings

```json
{
  "validation_scheme": {
    "type": "time_only",
    "time_folds": {"n_folds": 2, "train_months": 12, "val_months": 3}
  },
  "optimization_settings": {
    "iters_per_fold": 5000,
    "population_size": 50
  }
}
```

### Production Settings

```json
{
  "validation_scheme": {
    "type": "combined",
    "time_folds": {"n_folds": 3, "train_months": 18, "val_months": 6},
    "coin_folds": {"n_folds": 2}
  },
  "optimization_settings": {
    "iters_per_fold": 20000,
    "population_size": 150
  }
}
```

See `meta_optimizer/configs/` for example configurations.

---

## Troubleshooting

### "venv not found"

Run deploy first:
```bash
./meta_optimizer/deploy.sh deploy
```

### "Rust not found"

Deploy will auto-install Rust, or run setup:
```bash
./meta_optimizer/deploy.sh setup
```

### "Permission denied"

Check your SSH key in `.env`:
```env
SSH_KEY=~/.ssh/your-key.pem
```

### View detailed logs

```bash
./meta_optimizer/deploy.sh logs --lines 500
```

### Resume after interruption

Progress is auto-saved. Results continue from last checkpoint.

---

## Example Workflow

```bash
# 1. Setup (first time only)
cp meta_optimizer/.env.example meta_optimizer/.env
nano meta_optimizer/.env  # Edit with your details
./meta_optimizer/deploy.sh setup

# 2. Deploy code
./meta_optimizer/deploy.sh deploy

# 3. Test installation
./meta_optimizer/deploy.sh test

# 4. Run quick test first
./meta_optimizer/deploy.sh run --quick-test

# 5. Monitor in another terminal
./meta_optimizer/deploy.sh monitor

# 6. When done, download results
./meta_optimizer/deploy.sh download --output ./results

# 7. Check best configs
cat ./results/*/best_configs/rank_1.json
```

---

## Files Reference

| File | Description |
|------|-------------|
| `deploy.sh` | Main deployment & management script |
| `monitor_dashboard.py` | Local visual monitoring dashboard |
| `cli.py` | Meta-optimizer CLI |
| `orchestrator.py` | Main coordination logic |
| `use.sh` | Interactive usage guide |
| `.env` | Your configuration (create from .env.example) |
| `configs/default_meta_config.json` | Default settings |
| `configs/quick_test.json` | Fast test settings |
