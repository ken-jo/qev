"""Paired group-bootstrap accuracy comparison of two frozen evaluation reports."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(a, b):
    if a["subset_sha256"] != b["subset_sha256"] or a["records_sha256"] != b["records_sha256"]:
        raise ValueError("reports do not use identical records and split")

    def key(r):
        return (r["id"], r["question"])

    aa, bb = {key(r): r for r in a["rows"]}, {key(r): r for r in b["rows"]}
    if aa.keys() != bb.keys():
        raise ValueError("question populations differ")
    groups = {}
    for k, r in aa.items():
        if r["group"] != bb[k]["group"]:
            raise ValueError("group identities differ")
        groups.setdefault(r["group"], []).append(r["correct"] - bb[k]["correct"])
    values = list(groups.values())
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(values), size=(2000, len(values)))
    sums = np.array([sum(v) for v in values])
    counts = np.array([len(v) for v in values])
    delta = sums[draws].sum(1) / counts[draws].sum(1)
    return {
        "pemli": a["summary"],
        "qev08": b["summary"],
        "accuracy_delta": float(sums.sum() / counts.sum()),
        "paired_group_bootstrap_95_interval": np.percentile(delta, [2.5, 97.5]).tolist(),
        "groups": len(groups),
        "scope": (
            "same test population; resident serial latency; "
            "no superiority claim from one point estimate"
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pemli", type=Path, required=True)
    p.add_argument("--qev08", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("comparison output is immutable")
    args.output.write_text(
        json.dumps(
            compare(json.loads(args.pemli.read_text()), json.loads(args.qev08.read_text())),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
