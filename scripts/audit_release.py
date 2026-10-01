"""Bind final, calibration, regression and latency evidence before checkpoint publication."""

import argparse
import hashlib
import json
from pathlib import Path

from veyra.release_gate import audit_release, evidence_hashes

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
for name in ("quality", "calibration", "integration", "regression", "protocol", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("release audits are immutable; use a new output path")
paths = {
    name: getattr(args, name)
    for name in ("quality", "calibration", "integration", "regression", "protocol")
}
manifest = args.checkpoint / "manifest.json"
result = audit_release(
    json.loads(manifest.read_text()),
    hashlib.sha256((args.checkpoint / "head.safetensors").read_bytes()).hexdigest(),
    hashlib.sha256(manifest.read_bytes()).hexdigest(),
    **{name: json.loads(path.read_text()) for name, path in paths.items()},
)
result["evidence"] = {
    name: {"path": str(path), **evidence_hashes(path.read_bytes())} for name, path in paths.items()
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
