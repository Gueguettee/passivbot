# Running Passivbot with Docker

This guide explains how to run Passivbot using Docker and Docker Compose. This approach ensures all dependencies (Python, Rust) are correctly installed in an isolated environment without modifying your local system.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your machine.
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop).

## Services Overview

The `docker-compose.yml` defines two services:

1.  **`passivbot`**: The general-purpose container. Use this for backtesting, optimization, or general tasks. It mounts the entire user directory into `/app`, so any changes to code or configs are immediately reflected.
2.  **`passivbot-live`**: A lightweight container optimized for live trading. It only mounts the `./configs/` directory and `api-keys.json`. This is safer for live environments as it isolates the core code.

## Quick Start

1.  **Build the images:**
    ```bash
    docker-compose build
    ```

## Usage Examples

### 1. Backtesting

To run a backtest, use the `passivbot` service. You can override the default command to run `src/backtest.py`.

```bash
# Run a backtest using a specific config
docker-compose run --rm passivbot python src/backtest.py configs/examples/top20mcap.json
```

*   `--rm`: Removes the container after it exits (keeps things clean).
*   `passivbot`: The service name from `docker-compose.yml`.
*   `python src/backtest.py ...`: The specific command to execute.

### 2. Optimization

Similarly, use the `passivbot` service for optimization.

```bash
docker-compose run --rm passivbot python src/optimize.py configs/template.json
```

### 3. Live Trading

For live trading, ensure you have your `api-keys.json` set up in the root directory.

1.  **Prepare API Keys:**
    Copy the example and add your keys:
    ```bash
    cp api-keys.json.example api-keys.json
    # Edit api-keys.json with your actual keys
    ```

2.  **Run Live Bot:**
    Use the `passivbot-live` profile/service.

    ```bash
    docker-compose --profile live run --rm passivbot-live python src/main.py configs/your_config.json
    ```

    *   Note: The `passivbot-live` service mounts `./configs` and `./api-keys.json`, so ensure your configuration file is inside the `configs/` folder.

### 4. Running in Background (Detached)

If you want the bot to keep running in the background (typical for live trading):

```bash
docker-compose --profile live up -d passivbot-live
```
*   **Note:** You might need to edit `docker-compose.yml` to set the specific command (e.g., `command: python src/main.py configs/my_config.json`) if you use `up -d`, or ensure the default command is what you want.

### 5. Interactive Shell

If you need to explore the environment or run multiple commands:

```bash
docker-compose run --rm passivbot bash
```
On Windows, if `bash` is not available or you are using PowerShell/CMD and encounter issues, you might try:
```bash
docker-compose run --rm passivbot /bin/sh
```

## Troubleshooting

*   **Permissions:** On Linux, if you encounter permission issues with created files (like backtest results), ensure your user ID matches the container user or adjust permissions afterward.
*   **Rebuild:** If you modify `requirements.txt` or `Dockerfile`, remember to rebuild:
    ```bash
    docker-compose build --no-cache
    ```
