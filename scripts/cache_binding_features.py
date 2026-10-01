"""Cache frozen, merged workspace features from training and development requests only."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import save_file

from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.interventions import augment_unit, retype_unit, training_units
from veyra.option_model import OptionModel
from veyra.replay import audit_replay, replay_pool, sample_replay


def controlled_training(records, rng):
    train = [record for record in records if record.split == "train"]
    result = []
    for unit in training_units(train, auxiliary=False):
        result.extend(augment_unit(retype_unit(unit, rng), rng))
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--replay-records", type=Path, required=True)
    parser.add_argument("--replay-count", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("feature output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_records(args.records)
    replay = [record for record in read_records(args.replay_records) if record.split == "train"]
    audit = audit_replay(records, args.records.parent, replay, args.replay_records.parent)
    rng = random.Random(args.seed)
    primary = controlled_training(records, rng)
    dev = [record for record in records if record.split == "dev"]
    replay = sample_replay(replay_pool(replay), args.replay_count, rng)
    if args.smoke:
        primary = primary[:8] + [r for r in primary if r.family.startswith("image_")][:8]
        dev = dev[:4] + [r for r in dev if r.family.startswith("image_")][:4]
        replay = replay[:6]
    items = [(record, args.records.parent) for record in primary]
    items += [(record, args.replay_records.parent) for record in replay]
    items += [(record, args.records.parent) for record in dev]
    if any(record.split not in {"train", "dev"} for record, _ in items):
        raise ValueError("binding features cannot contain calibration or final requests")
    parent_path = args.checkpoint / "manifest.json"
    parent = json.loads(parent_path.read_text())
    if not parent.get("reasoning_slots", 0) or parent.get("binding_head", {}).get("rank", 0):
        raise ValueError("extract from an unmodified workspace checkpoint")
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            Path("src/veyra/option_model.py"),
            Path("src/veyra/reasoning_workspace.py"),
            Path("src/veyra/interventions.py"),
            Path("src/veyra/replay.py"),
        )
    }
    specification = {
        "arguments": vars(args),
        "parent_manifest_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "parent_weights_sha256": parent["weights_sha256"],
        "dataset_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "replay_dataset_sha256": hashlib.sha256(args.replay_records.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "replay_audit": audit,
        "actual_requests_sha256": hashlib.sha256(
            "".join(record.model_dump_json() + "\n" for record, _ in items).encode()
        ).hexdigest(),
        "calibration_or_final_encoded": False,
        "smoke": args.smoke,
    }
    (args.output / "specification.json").write_text(
        json.dumps(specification, default=str, indent=2) + "\n", encoding="utf-8"
    )
    model = OptionModel.load(args.checkpoint, local_files_only=True, merge=True)
    states, conditions, base, targets, valid, metadata = [], [], [], [], [], []
    started = time.perf_counter()
    for record, root in items:
        if len(record.request.questions) != 1:
            raise ValueError("cache items must contain one question")
        hidden, mappings, _, condition_hidden = model.encode(
            record.request, root, capture_condition=True
        )
        name, question = next(iter(record.request.questions.items()))
        target, mask = torch.zeros(16), torch.zeros(16, dtype=torch.bool)
        for candidate, position in zip(candidates_for(question), mappings[name], strict=True):
            target[position] = record.targets[name][candidate.key]
            mask[position] = True
        states.append(hidden[0].cpu())
        conditions.append(model.condition_readout(condition_hidden)[0].cpu())
        base.append(model.readout(hidden)[0].cpu())
        targets.append(target)
        valid.append(mask)
        metadata.append(
            {
                "id": record.id,
                "group": record.group_id,
                "split": record.split,
                "family": record.family,
                "type": question.type,
                "language": record.language,
            }
        )
        if len(metadata) % 200 == 0 or len(metadata) == len(items):
            print(f"encoded {len(metadata)}/{len(items)} train/development questions", flush=True)
    tensors = {
        "states": torch.stack(states),
        "condition_logits": torch.stack(conditions),
        "base_logits": torch.stack(base),
        "targets": torch.stack(targets),
        "valid": torch.stack(valid),
    }
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("feature cache contains nonfinite values")
    weights = args.output / "features.safetensors"
    save_file(tensors, weights)
    metadata_path = args.output / "records.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "complete": True,
        "rows": len(metadata),
        "by_split": dict(Counter(row["split"] for row in metadata)),
        "features_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "specification_sha256": hashlib.sha256(
            (args.output / "specification.json").read_bytes()
        ).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
