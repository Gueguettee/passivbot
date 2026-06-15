# AGENTS.md

Instructions for AI coding assistants working on Passivbot.

## Always Read First

Read these files for every task:

1. `AGENTS.md`
2. `docs/ai/principles.yaml`
3. `docs/ai/error_contract.md`

Then use `docs/ai/README.md` to load task-specific docs only when relevant.

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest
passivbot live -u {account_name}
```

Legacy direct-script entrypoints such as `python3 src/main.py ...` still work, but prefer the
unified `passivbot ...` CLI for new usage.

## Non-Negotiables

1. Rust is source of truth for order behavior.
- Behavior changes in entries/closes/risk/unstuck belong in `passivbot-rust/src/`, not Python patches.
2. Stateless behavior is required.
- Bot behavior must be reproducible after restart from exchange state + config.
3. Fail loudly in trading-critical paths.
- Default is hard-fail for exchange data, EMA inputs, risk gates, and order construction.
- Fallbacks are exceptions, not defaults.
- See `docs/ai/error_contract.md` for the full fallback matrix.
4. Keep terminology and signed-qty conventions exact.
- `position_side` = long/short.
- `side` / `order_side` = buy/sell.
- `qty` and `pos_size` are signed in internal logic.
5. EMA spans are floats.
- Do not round derived spans like `sqrt(span0 * span1)`.
6. Avoid scope creep.
- Make only requested or strictly necessary changes.

## Before Coding

1. Read `docs/ai/README.md` and open only docs relevant to the task.
2. If touching exchange code, read `docs/ai/exchange_api_quirks.md`.
3. If touching Rust/PyO3 packaging or tests, read `docs/ai/build_pitfalls.md`.
4. If touching a documented feature, read the corresponding file in `docs/ai/features/`.
5. Check branch context before broad edits:

```bash
git branch --show-current
git log --oneline -n 10
```

6. Run a silent-handling self-audit for touched areas:

```bash
rg -n "except Exception|return_exceptions=True|\.get\([^\n]*,\s*(0|0\.0|None|False|\{\}|\[\])\)" src tests
```

7. Remove unsafe patterns or document explicit, approved fallback behavior with tests.

## Testing Expectations

1. Run targeted tests for changed paths.
2. Add regression tests for bug fixes and fallback behavior.
3. If Rust changed, rebuild extension before Python tests:

```bash
cd passivbot-rust && maturin develop --release && cd ..
```

See `docs/ai/code_review_prompt.md` for the review/test checklist.

## Commands

Use `docs/ai/commands.md` for setup, test, backtest, optimizer, and Rust build commands.

## Live Deployment (this fork)

Live bots run as Docker containers on a remote VM (Oracle Cloud ARM), one container
per exchange account. Code is **not** version-controlled on the box — it is an rsynced
copy. The bot code is baked into the image at build time (`Dockerfile_live`); only
`configs/` and `api-keys.json` are bind-mounted, so any code change requires a rebuild.

`deploy.sh` (repo root) drives the whole lifecycle. It reads remote host/user/key from
`.env` (`REMOTE_USER`, `REMOTE_HOST`, `SSH_KEY_PATH`, `REMOTE_DIR`). Run it from a shell
where the SSH key is reachable (on Windows the key lives in WSL at
`~/.ssh/id_rsa_oracle`, so invoke via `wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/c/git/passivbot && ./deploy.sh ...'`).

Two live targets, selected with `--hl`:

| Target      | Flag   | Container          | Account          | Config                                                  |
| ----------- | ------ | ------------------ | ---------------- | ------------------------------------------------------- |
| Binance     | (none) | `passivbot-live`   | `binance_01`     | `configs/original/hype_dio_masterclas_binance_opti/...` |
| Hyperliquid | `--hl` | `passivbot-live-hl`| `hyperliquid_01` | `configs/original/hype_dio_masterclass/...`             |

Commands (append `--hl` to target Hyperliquid):

```bash
./deploy.sh deploy      # rsync code -> build image -> recreate container -> tail 30s
./deploy.sh restart     # stop then deploy
./deploy.sh stop        # docker compose down for that profile
./deploy.sh logs -f     # follow logs (-n NUM for tail count)
./deploy.sh status      # container ps + resource usage + disk
./deploy.sh connect     # ssh into the box at REMOTE_DIR
```

To relaunch **both** running configs after a code change: `./deploy.sh deploy` then
`./deploy.sh deploy --hl`. The second build is layer-cached. The compose profiles are
independent, so each deploy only recreates its own container.

Caveats:

- `sync_to_remote()` uses `rsync --delete`: remote files absent locally are removed.
  It excludes `.git`, `venv`, caches, `meta_optimizer`, `optimize_results`, `backtests*`,
  `.env`, `*.log`, etc. — but **not** `meta_optimize_results/`, so optimizer outputs left
  on the box get deleted on deploy. Pull anything you want to keep before deploying.
- Before a rebuild on the shared box, sanity-check resources (`free -h`, `swapon --show`,
  `uptime`, `df -h /`) — it also runs the user's gueguetteBot and a polymarket bot.
- After deploy, confirm health: `docker ps --filter name=passivbot-live` and look for a
  `[health]` log line (`errors=0/10`, expected positions/balance) plus an `[order] wave
  complete` line on each container.

## Documentation Hygiene

1. Keep AI docs lean and task-oriented.
2. Put durable rules in `principles.yaml` or `error_contract.md`, not in many files.
3. Put deep investigations in case-study docs, not core instruction docs.
4. User-facing docs and `CHANGELOG.md` should describe the diff from `master`, not intermediate changes made within the current dev branch.
5. Update `CHANGELOG.md` for user-facing behavior changes.
