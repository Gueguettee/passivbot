# Launch backtest:
docker compose run --rm passivbot python src/backtest.py strategies/hype_top.json
docker compose run --rm passivbot python src/backtest.py strategies/hype_top_4pairs.json

# Launch backtest in detached mode:
docker compose run -d passivbot python src/backtest.py strategies/hype_top.json
docker compose run -d passivbot python src/backtest.py strategies/hype_top_4pairs.json

# Launch optimization:
docker compose run --rm passivbot python src/optimize.py strategies/hype_top_4pairs.json

# Launch Pareto dashboard:
docker compose run -p 8050:8050 -e HOST=0.0.0.0 passivbot python src/tools/pareto_dash.py --data-root optimize_results

# Check for empty optimization result files:
docker compose run passivbot find optimize_results -name "*.json" -size 0
