"""Build a new controlled policy corpus; existing directories are never overwritten."""

import argparse
import json
from pathlib import Path

from veyra.policy_data import build_policy_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--scale", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build_policy_data(args.output, args.seed, args.scale), indent=2))


if __name__ == "__main__":
    main()
