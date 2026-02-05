#!/usr/bin/env python3
"""
Meta-Optimizer Remote Monitoring Dashboard

A real-time dashboard for monitoring meta-optimization runs on remote instances.

Configuration is loaded from .env file in the meta_optimizer directory.

Usage:
    python monitor_dashboard.py
    python monitor_dashboard.py --refresh 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def load_env_file() -> Dict[str, str]:
    """Load configuration from .env file."""
    env_vars = {}
    script_dir = Path(__file__).parent
    env_file = script_dir / ".env"

    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value and value[0] in ('"', "'") and value[-1] == value[0]:
                        value = value[1:-1]
                    env_vars[key] = value
    else:
        print(f"Warning: .env file not found at {env_file}")
        print("Copy .env.example to .env and configure your settings.")

    return env_vars


# ANSI color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def ssh_command(host: str, cmd: str, key: Optional[str] = None) -> Tuple[str, int]:
    """Execute SSH command and return output."""
    ssh_cmd = ["ssh"]
    if key:
        ssh_cmd.extend(["-i", key])
    ssh_cmd.extend(["-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no"])
    ssh_cmd.append(host)
    ssh_cmd.append(cmd)

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "Timeout", 1
    except Exception as e:
        return str(e), 1


def get_system_stats(host: str, key: Optional[str]) -> Dict[str, Any]:
    """Get system statistics from remote host."""
    stats = {}

    # CPU usage
    cpu_cmd = "grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage}'"
    cpu_out, _ = ssh_command(host, cpu_cmd, key)
    try:
        stats['cpu_percent'] = float(cpu_out)
    except:
        stats['cpu_percent'] = 0

    # Memory usage
    mem_cmd = "free -m | awk 'NR==2{printf \"%s %s %.1f\", $3, $2, $3*100/$2}'"
    mem_out, _ = ssh_command(host, mem_cmd, key)
    try:
        parts = mem_out.split()
        stats['mem_used_mb'] = int(parts[0])
        stats['mem_total_mb'] = int(parts[1])
        stats['mem_percent'] = float(parts[2])
    except:
        stats['mem_used_mb'] = 0
        stats['mem_total_mb'] = 0
        stats['mem_percent'] = 0

    # Disk usage
    disk_cmd = "df -h / | awk 'NR==2{print $3, $2, $5}'"
    disk_out, _ = ssh_command(host, disk_cmd, key)
    try:
        parts = disk_out.split()
        stats['disk_used'] = parts[0]
        stats['disk_total'] = parts[1]
        stats['disk_percent'] = parts[2]
    except:
        stats['disk_used'] = '?'
        stats['disk_total'] = '?'
        stats['disk_percent'] = '?'

    # Uptime
    uptime_cmd = "uptime -p"
    uptime_out, _ = ssh_command(host, uptime_cmd, key)
    stats['uptime'] = uptime_out

    return stats


def get_process_status(host: str, key: Optional[str], remote_dir: str) -> Dict[str, Any]:
    """Get meta-optimizer process status."""
    status = {
        'running': False,
        'pid': None,
        'runtime': None,
    }

    # Check tmux session
    tmux_cmd = "tmux has-session -t meta_opt 2>/dev/null && echo 'running' || echo 'stopped'"
    tmux_out, _ = ssh_command(host, tmux_cmd, key)
    status['running'] = 'running' in tmux_out

    # Get PID if running
    if status['running']:
        pid_cmd = "pgrep -f 'meta_optimizer/cli.py' | head -1"
        pid_out, _ = ssh_command(host, pid_cmd, key)
        if pid_out.isdigit():
            status['pid'] = int(pid_out)

            # Get process start time
            time_cmd = f"ps -o etimes= -p {status['pid']} 2>/dev/null"
            time_out, _ = ssh_command(host, time_cmd, key)
            try:
                seconds = int(time_out.strip())
                status['runtime'] = str(timedelta(seconds=seconds))
            except:
                pass

    return status


def get_optimization_progress(host: str, key: Optional[str], remote_dir: str) -> Dict[str, Any]:
    """Get optimization progress from checkpoint."""
    progress = {
        'phase': 'unknown',
        'results_dir': None,
        'folds_total': 0,
        'folds_completed': 0,
        'configs_found': 0,
        'configs_validated': 0,
        'validations_total': 0,
        'validations_done': 0,
        'best_score': None,
    }

    # Find latest results directory
    find_cmd = f"ls -td {remote_dir}/meta_optimize_results/*/ 2>/dev/null | head -1"
    dir_out, rc = ssh_command(host, find_cmd, key)

    if rc != 0 or not dir_out:
        return progress

    progress['results_dir'] = Path(dir_out).name

    # Read checkpoint
    checkpoint_cmd = f"cat {dir_out}/checkpoint.json 2>/dev/null"
    checkpoint_out, rc = ssh_command(host, checkpoint_cmd, key)

    if rc != 0:
        return progress

    try:
        checkpoint = json.loads(checkpoint_out)
        state = checkpoint.get('state', {})

        progress['phase'] = state.get('phase', 'unknown')

        # Folds
        folds = state.get('folds', [])
        progress['folds_total'] = len(folds)

        opt_results = state.get('optimization_results', {})
        progress['folds_completed'] = len(opt_results)

        # Configs
        all_configs = state.get('all_configs', {})
        progress['configs_found'] = len(all_configs)

        # Validations
        val_matrix = state.get('validation_matrix', {})
        progress['configs_validated'] = len(val_matrix)

        if val_matrix:
            total_vals = sum(len(v) for v in val_matrix.values())
            expected_vals = len(all_configs) * progress['folds_total']
            progress['validations_done'] = total_vals
            progress['validations_total'] = expected_vals

        # Best score
        scores = state.get('robustness_scores', {})
        if scores:
            best = max(scores.values(), key=lambda x: x.get('overall_score', 0))
            progress['best_score'] = best.get('overall_score')

    except json.JSONDecodeError:
        pass

    return progress


def get_recent_logs(host: str, key: Optional[str], remote_dir: str, lines: int = 5) -> str:
    """Get recent log lines."""
    log_cmd = f"ls -t {remote_dir}/logs/meta_opt_*.log 2>/dev/null | head -1 | xargs tail -n {lines} 2>/dev/null"
    log_out, _ = ssh_command(host, log_cmd, key)
    return log_out if log_out else "No logs available"


def format_bar(percent: float, width: int = 30) -> str:
    """Format a progress bar."""
    filled = int(width * percent / 100)
    empty = width - filled

    if percent < 50:
        color = Colors.GREEN
    elif percent < 80:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    bar = f"{color}{'█' * filled}{'░' * empty}{Colors.RESET}"
    return f"[{bar}] {percent:.1f}%"


def print_header():
    """Print dashboard header."""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           META-OPTIMIZER MONITORING DASHBOARD                  ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(Colors.RESET)


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}═══ {title} ═══{Colors.RESET}")


def run_dashboard(host: str, key: Optional[str], remote_dir: str, refresh: int = 5):
    """Run the monitoring dashboard."""
    print(f"Connecting to {host}...")
    print("Press Ctrl+C to exit\n")

    while True:
        try:
            clear_screen()
            print_header()
            print(f"Host: {Colors.CYAN}{host}{Colors.RESET}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # System stats
            print_section("SYSTEM RESOURCES")
            sys_stats = get_system_stats(host, key)

            print(f"  CPU:    {format_bar(sys_stats['cpu_percent'])}")
            print(f"  Memory: {format_bar(sys_stats['mem_percent'])} "
                  f"({sys_stats['mem_used_mb']}MB / {sys_stats['mem_total_mb']}MB)")
            print(f"  Disk:   {sys_stats['disk_used']} / {sys_stats['disk_total']} ({sys_stats['disk_percent']})")
            print(f"  Uptime: {sys_stats['uptime']}")

            # Process status
            print_section("PROCESS STATUS")
            proc_status = get_process_status(host, key, remote_dir)

            if proc_status['running']:
                status_str = f"{Colors.GREEN}● RUNNING{Colors.RESET}"
            else:
                status_str = f"{Colors.RED}○ STOPPED{Colors.RESET}"

            print(f"  Status:  {status_str}")
            if proc_status['pid']:
                print(f"  PID:     {proc_status['pid']}")
            if proc_status['runtime']:
                print(f"  Runtime: {proc_status['runtime']}")

            # Optimization progress
            print_section("OPTIMIZATION PROGRESS")
            progress = get_optimization_progress(host, key, remote_dir)

            if progress['results_dir']:
                print(f"  Results Dir: {progress['results_dir']}")
                print(f"  Phase:       {Colors.YELLOW}{progress['phase']}{Colors.RESET}")
                print()

                # Folds progress
                if progress['folds_total'] > 0:
                    fold_pct = 100 * progress['folds_completed'] / progress['folds_total']
                    print(f"  Optimization Folds: {progress['folds_completed']}/{progress['folds_total']}")
                    print(f"  {format_bar(fold_pct, 40)}")

                # Configs found
                print(f"\n  Configs Found:     {Colors.CYAN}{progress['configs_found']}{Colors.RESET}")
                print(f"  Configs Validated: {progress['configs_validated']}")

                # Validations
                if progress['validations_total'] > 0:
                    val_pct = 100 * progress['validations_done'] / progress['validations_total']
                    print(f"\n  Validations: {progress['validations_done']}/{progress['validations_total']}")
                    print(f"  {format_bar(val_pct, 40)}")

                # Best score
                if progress['best_score'] is not None:
                    score_color = Colors.GREEN if progress['best_score'] > 0.5 else Colors.YELLOW
                    print(f"\n  Best Robustness Score: {score_color}{progress['best_score']:.4f}{Colors.RESET}")
            else:
                print(f"  {Colors.YELLOW}No optimization results found yet{Colors.RESET}")

            # Recent logs
            print_section("RECENT LOGS")
            logs = get_recent_logs(host, key, remote_dir, lines=8)
            for line in logs.split('\n')[-8:]:
                if line.strip():
                    # Colorize log levels
                    if 'ERROR' in line:
                        print(f"  {Colors.RED}{line}{Colors.RESET}")
                    elif 'WARN' in line:
                        print(f"  {Colors.YELLOW}{line}{Colors.RESET}")
                    elif 'INFO' in line:
                        print(f"  {Colors.WHITE}{line}{Colors.RESET}")
                    else:
                        print(f"  {line}")

            # Footer
            print(f"\n{Colors.CYAN}─" * 68 + Colors.RESET)
            print(f"Refreshing every {refresh}s | Press Ctrl+C to exit")
            print(f"To attach: ssh {host} -t 'tmux attach -t meta_opt'")

            time.sleep(refresh)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Dashboard stopped.{Colors.RESET}")
            break
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
            time.sleep(refresh)


def main():
    parser = argparse.ArgumentParser(
        description="Meta-Optimizer Remote Monitoring Dashboard",
        epilog="Configuration (REMOTE_HOST, SSH_KEY) is loaded from .env file."
    )
    parser.add_argument(
        "--refresh", "-r",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)"
    )

    args = parser.parse_args()

    # Load configuration from .env file
    env_config = load_env_file()

    host = env_config.get('REMOTE_HOST')
    key = env_config.get('SSH_KEY')
    remote_dir = env_config.get('REMOTE_DIR', '/home/ubuntu/passivbot')

    # Expand ~ in paths
    if key and key.startswith('~'):
        key = os.path.expanduser(key)

    if not host:
        print(f"{Colors.RED}Error: REMOTE_HOST not configured.{Colors.RESET}")
        print("Please set REMOTE_HOST in .env file")
        print("")
        print("To create .env file:")
        script_dir = Path(__file__).parent
        print(f"  cp {script_dir}/.env.example {script_dir}/.env")
        sys.exit(1)

    run_dashboard(
        host=host,
        key=key,
        remote_dir=remote_dir,
        refresh=args.refresh
    )


if __name__ == "__main__":
    main()
