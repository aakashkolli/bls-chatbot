"""
swap_model_config.py — Update the MODEL variable in .env

Usage:
    python swap_model_config.py --model "deepseek-r1:32b"
    python swap_model_config.py --show
"""

import argparse
import os
import re
from pathlib import Path

ENV_FILE = Path(__file__).parent.parent / ".env"


def read_env() -> dict:
    """Parse .env into a dict."""
    values = {}
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    values[key.strip()] = val.strip()
    return values


def write_model(model_id: str) -> None:
    """Replace the MODEL= line in .env with the new model identifier."""
    if not ENV_FILE.exists():
        raise FileNotFoundError(f".env file not found at {ENV_FILE}")

    content = ENV_FILE.read_text()
    new_line = f"MODEL={model_id}"

    if re.search(r"^MODEL=", content, re.MULTILINE):
        content = re.sub(r"^MODEL=.*$", new_line, content, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\n{new_line}\n"

    ENV_FILE.write_text(content)
    print(f"[swap_model_config] MODEL set to: {model_id}")


def show_current() -> None:
    values = read_env()
    print(f"Current MODEL in .env: {values.get('MODEL', '<not set>')}")


def main():
    parser = argparse.ArgumentParser(description="Swap MODEL in .env")
    parser.add_argument("--model", type=str, help="Model identifier to write to .env")
    parser.add_argument("--show",  action="store_true", help="Show current MODEL value")
    args = parser.parse_args()

    if args.show or not args.model:
        show_current()
    if args.model:
        write_model(args.model)


if __name__ == "__main__":
    main()
