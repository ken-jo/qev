import argparse
import json
from pathlib import Path

from veyra.evidence_data import build_evidence_data

parser = argparse.ArgumentParser()
parser.add_argument("--parent", type=Path, default=Path("data/policy-v3/records.jsonl"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=20261003)
args = parser.parse_args()
print(json.dumps(build_evidence_data(args.parent, args.output, args.seed), indent=2))
