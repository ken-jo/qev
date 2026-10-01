"""Audit eligible depth selection before running any fresh calibration inference."""

import argparse
import json
from pathlib import Path

from depth_release_checks import collect_depth_study, digest
from train_foundation_head import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("development audit is immutable")
    result = collect_depth_study(args.selection, args.records)
    result.update(
        passed=True,
        source_sha256=digest(Path(__file__)),
        selection_sha256=digest(args.selection),
        fresh_calibration_or_final_inference_used=False,
        release_allowed=False,
    )
    write(args.output, result)
    print(json.dumps({"passed": True, "release_allowed": False}), flush=True)


if __name__ == "__main__":
    main()
