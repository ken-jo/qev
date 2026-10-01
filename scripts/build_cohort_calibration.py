"""Combine previously inspected calibration populations; preserve all other records."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from veyra.data import read_records


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config_path = Path("configs/workflow-cohort-policy-v12.json")
    config = read(config_path)
    design_path = Path(config["fitting_data_design"])
    if digest(design_path) != config["fitting_data_design_sha256"]:
        raise ValueError("Combined fitting design changed")
    design = read(design_path)
    root = Path(design["output"])
    if root.exists():
        raise FileExistsError("Combined fitting data is immutable")
    original_path = Path(design["original_records"])
    original = read_records(original_path)
    records = [r for r in original if r.split != "calibration"]
    populations = [(original_path, records)]
    cohort_groups = {}
    for cohort, spec in config["fitting_populations"].items():
        path = Path(spec["records"])
        if digest(path) != spec["records_sha256"]:
            raise ValueError("Source calibration population changed")
        selected = [r for r in read_records(path) if r.split == "calibration"]
        if len(selected) != 2644:
            raise ValueError("Unexpected fitting population")
        groups = {r.group_id for r in selected}
        if any(groups & previous for previous in cohort_groups.values()):
            raise ValueError("Fitting cohorts overlap")
        cohort_groups[cohort] = groups
        records.extend(selected)
        populations.append((path, selected))
    if len({r.id for r in records}) != len(records):
        raise ValueError("Combined record identifiers overlap")
    root.mkdir(parents=True)
    for path, population in populations:
        for record in population:
            for image in record.request.state.images:
                source, target = path.parent / image.path, root / image.path
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Image path escapes combined data folder")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if digest(source) != digest(target):
                        raise ValueError("Conflicting image names")
                else:
                    os.link(source, target)
    output = root / "records.jsonl"
    output.write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8", newline="\n"
    )
    split_hashes = {}
    for split in ("train", "dev", "test"):
        before = "".join(r.model_dump_json() + "\n" for r in original if r.split == split)
        after = "".join(r.model_dump_json() + "\n" for r in records if r.split == split)
        if before != after:
            raise ValueError("Non-calibration records changed")
        split_hashes[split] = hashlib.sha256(before.encode()).hexdigest()
    result = {
        "passed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_sha256": digest(config_path),
        "design_sha256": digest(design_path),
        "source_sha256": digest(__file__),
        "records_sha256": digest(output),
        "cohorts": {key: len(value) for key, value in cohort_groups.items()},
        "fitting_questions": 7932,
        "fitting_groups": sum(map(len, cohort_groups.values())),
        "preserved_splits": split_hashes,
        "previous_validation_now_fitting_data": True,
        "final_predictions_used": False,
        "release_allowed": False,
    }
    (root / "protocol.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
