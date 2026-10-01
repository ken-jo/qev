"""Preserve the selected depth weights after auditing fresh calibration separation."""

import json
from pathlib import Path

import torch
from cohort_policy_checks import collect_cohort_design, expected_selection
from depth_policy_checks import digest


def main():
    torch.set_num_threads(4)
    root = Path("runs/workflow-v12-cohort-policy-selection")
    if root.exists():
        raise FileExistsError("policy selection preparation is immutable")
    design = collect_cohort_design()
    selection = expected_selection(design)
    audit = {
        "passed": True,
        "source_sha256": digest(__file__),
        "weights_sha256": selection["weights_sha256"],
        "candidate_weights_reselected": False,
        "data_separation": design["data_separation"],
        "evidence_files": design["evidence_files"],
        "source_files": design["source_files"],
        "model_predictions_used": False,
        "release_allowed": False,
    }
    root.mkdir(parents=True)
    for name, value in (("selection.json", selection), ("data-and-lineage-audit.json", audit)):
        (root / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "prepared": True,
                "data_separation": design["data_separation"],
                "release_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
