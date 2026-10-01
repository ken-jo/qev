"""Publish completed epoch aggregates while keeping interim and merged evidence distinct."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    config_path = Path("configs/workflow-curriculum-study-v12.json")
    config = read(config_path)
    source = Path("runs/workflow-v12-curriculum-study")
    output = Path("reports/workflow-v12/skill-curriculum")
    index_path = output / "manifest.json"
    index = read(index_path)
    exported = []
    for arm in config["arms"]:
        protocol_path = source / arm / "protocol.json"
        if not protocol_path.exists():
            continue
        protocol = read(protocol_path)
        if protocol["configuration_sha256"] != digest(config_path):
            raise ValueError("study declaration changed")
        for name, expected in protocol["source_sha256"].items():
            if digest(Path(name)) != expected:
                raise ValueError("training source changed: " + name)
        for epoch in range(1, config["epochs"] + 1):
            checkpoint = source / arm / f"epoch-{epoch}" / "checkpoint"
            manifest_path = checkpoint / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = read(manifest_path)
            training = manifest["training"]
            weights = digest(checkpoint / "head.safetensors")
            if not (
                training["completed"] is True
                and training["intermediate"] is True
                and training["epochs"] == epoch
                and training["protocol_sha256"] == digest(protocol_path)
                and manifest["weights_sha256"] == weights
                and training["final_evaluated_at_selection"] is False
            ):
                raise ValueError("epoch evidence does not match its checkpoint")
            report = {
                "status": "interim_unmerged_development",
                "arm": arm,
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "weights_sha256": weights,
                "manifest_sha256": digest(manifest_path),
                "study_protocol_sha256": digest(config_path),
                "source_sha256": digest(Path(__file__)),
                "training_protocol": protocol,
                "training": training,
                "release_allowed": False,
                "limitation": (
                    "Repeatedly inspected development data and one training seed. "
                    "Merged selection, calibration and final evaluation remain separate. "
                    "Skill-query metrics cannot substitute for original release criteria."
                ),
            }
            target = output / f"{arm}-epoch-{epoch}.json"
            if target.exists():
                if read(target) != report:
                    raise ValueError("published epoch aggregate changed: " + str(target))
            else:
                write(target, report)
            index[target.name] = digest(target)
            domains = training["selected_dev"]["by_domain"]
            exported.append(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "new_accuracy": domains["workflow_new"]["accuracy"],
                    "known_accuracy": domains["workflow_known"]["accuracy"],
                    "uncertainty_nll": domains["uncertainty"]["nll"],
                    "uncertainty_cost_at_80pct": domains["uncertainty"]["matched_coverage"]["0.8"][
                        "expected_cost"
                    ],
                    "coverage_at_15pct_error": {
                        kind: row["maximum_coverage_at_15pct_error"]
                        for kind, row in training["development_risk_diagnostic"].items()
                    },
                    "skill_hard_accuracy": {
                        kind: training["skill_diagnostic"][kind]["hard_accuracy"]
                        for kind in ("skill_fact", "skill_rule")
                    },
                }
            )
    if not exported:
        raise ValueError("no completed curriculum epoch to export yet")
    write(index_path, dict(sorted(index.items())))
    print(json.dumps({"exported": exported, "release_allowed": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
