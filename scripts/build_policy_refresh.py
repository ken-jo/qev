"""Freeze a larger independent evaluation before the next model-selection cycle."""

import argparse
import json
from pathlib import Path

from veyra.policy_refresh import build_refresh

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("data/policy-v8"))
parser.add_argument("--exclude", type=Path, nargs="+", required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
if args.report.exists():
    raise FileExistsError("frozen protocol reports are immutable")
report = build_refresh(args.output, args.exclude)
args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {key: value for key, value in report.items() if key != "duplicate_groups_removed"}, indent=2
    )
)
