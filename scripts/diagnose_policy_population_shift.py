"""Preserve the observed failed validation, without fitting a replacement policy."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path("runs/workflow-v12-depth-policy-release")
    output = root / "population-shift-diagnosis.json"
    if output.exists():
        raise FileExistsError("Diagnosis is immutable")
    result = {"source_sha256": digest(__file__), "populations": {}, "evidence_files": {}}
    for name in ("calibration", "policy-validation"):
        path = root / name / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        result["evidence_files"][str(path)] = digest(path)
        by_type = {}
        for kind in ("choice", "score", "noul"):
            selected = [row for row in rows if row["type"] == kind]
            by_type[kind] = {}
            for domain in ["all", *sorted({row["domain"] for row in selected})]:
                subset = (
                    selected
                    if domain == "all"
                    else [row for row in selected if row["domain"] == domain]
                )
                accepted = [row for row in subset if not row["abstained"]]
                by_type[kind][domain] = {
                    "questions": len(subset),
                    "groups": len({row["group"] for row in subset}),
                    "accepted": len(accepted),
                    "coverage": len(accepted) / len(subset),
                    "expected_error": sum(row["expected_error"] for row in accepted) / len(accepted)
                    if accepted
                    else None,
                }
        result["populations"][name] = by_type
    result.update(
        policy_refitted=False,
        model_weights_changed=False,
        final_predictions_used=False,
        limitation="Post-hoc subgroup diagnosis; differences do not establish a causal mechanism.",
        observed_failure="Noul validation error exceeded 15%; final inference never started.",
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"preserved": str(output), "policy_refitted": False}))


if __name__ == "__main__":
    main()
