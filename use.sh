# Launch backtest:
docker compose run --rm passivbot python src/backtest.py strategies/hype_top.json
docker compose run --rm passivbot python src/backtest.py strategies/hype_top_4pairs.json

# Launch backtest in detached mode:
docker compose run -d passivbot python src/backtest.py strategies/hype_top.json
docker compose run -d passivbot python src/backtest.py strategies/hype_top_4pairs.json

# Launch optimization:
docker compose run --rm passivbot python src/optimize.py strategies/hype_top_4pairs.json