"""Cache declared train/dev groups with the frozen current backbone, dev first."""

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from train_foundation_head import write

from veyra.candidates import candidates_for
from veyra.constants import QUESTION_TYPES
from veyra.data import read_records
from veyra.option_model import OptionModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(config):
    items, reserved, training = [], set(), set()
    for corpus, spec in config["datasets"].items():
        path = Path(spec["path"])
        if digest(path) != spec["sha256"]:
            raise ValueError("Dataset changed: " + corpus)
        records = read_records(path)
        reserved.update(r.group_id for r in records if r.split != "train")
        families = defaultdict(lambda: defaultdict(list))
        for record in records:
            if record.split == "train":
                families[record.family][record.group_id].append(record)
            elif record.split == "dev":
                items.append((corpus, record, path.parent))
        for groups in families.values():
            selected = sorted(
                groups,
                key=lambda group: hashlib.sha256(
                    f"{config['seed']}:{corpus}:{group}".encode()
                ).hexdigest(),
            )[: spec["maximum_training_groups_per_family"]]
            for group in selected:
                training.add(group)
                items.extend((corpus, r, path.parent) for r in groups[group])
    if training & reserved:
        raise ValueError("Training groups overlap development/calibration/final groups")
    train_requests, dev_requests = set(), set()
    train_images, dev_images = set(), set()
    for _, row, _ in items:
        target = train_requests if row.split == "train" else dev_requests
        target.add(json.dumps(row.request.model_dump(mode="json"), sort_keys=True))
        if row.image_sha256:
            (train_images if row.split == "train" else dev_images).add(row.image_sha256)
    if train_requests & dev_requests or train_images & dev_images:
        raise ValueError("Exact requests or images overlap training and development")
    return sorted(items, key=lambda item: (item[1].split != "dev", item[0], item[1].id))


@torch.inference_mode()
def main():
    config_path = Path("configs/workflow-readout-recovery-v13.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path("data/features-readout-recovery-v13")
    if output.exists():
        raise FileExistsError("Recovery feature output is immutable")
    parent = Path(config["parent"])
    for name in ("weights", "manifest"):
        filename = "head.safetensors" if name == "weights" else "manifest.json"
        if digest(parent / filename) != config["parent_" + name + "_sha256"]:
            raise ValueError("Declared parent changed")
    items = prepare(config)
    source_paths = [Path(__file__), *Path("src/veyra").glob("*.py")]
    specification = {
        "design": str(config_path),
        "design_sha256": digest(config_path),
        "parent_weights_sha256": config["parent_weights_sha256"],
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "source_files": {str(p): digest(p) for p in source_paths},
        "selected_requests_sha256": hashlib.sha256(
            "".join(corpus + ":" + r.model_dump_json() + "\n" for corpus, r, _ in items).encode()
        ).hexdigest(),
        "counts": dict(Counter(corpus + "/" + r.split for corpus, r, _ in items)),
        "calibration_or_final_encoded": False,
        "train_groups_disjoint_from_all_reserved_groups": True,
        "exact_train_dev_request_and_image_overlap": False,
    }
    write(output / "specification.json", specification)
    model = OptionModel.load(parent, local_files_only=True, merge=True)
    started = time.perf_counter()
    completed = 0
    split_manifests = {}
    for split in ("dev", "train"):
        collected = defaultdict(list)
        rows = []
        for corpus, record, root in items:
            if record.split != split:
                continue
            if len(record.request.questions) != 1:
                raise ValueError("Require one question per cache row")
            states, mappings, tokens, conditions = model.encode(
                record.request, root, capture_condition=True
            )
            condition_logits = model.condition_readout(conditions)
            logits = model.readout(states) + model.binding_head(states, condition_logits)
            name, question = next(iter(record.request.questions.items()))
            mapping = dict(
                zip((c.key for c in candidates_for(question)), mappings[name], strict=True)
            )
            target, valid = torch.zeros(16), torch.zeros(16, dtype=torch.bool)
            for key, position in mapping.items():
                target[position] = record.targets[name][key]
                valid[position] = True
            gold_tags = [t[9:] for t in record.tags if t.startswith("gold_key:")]
            if len(gold_tags) > 1:
                raise ValueError("Multiple hard labels")
            gold = mapping[gold_tags[0]] if gold_tags else -1
            if corpus == "legacy":
                if target.max().item() != 1 or target.sum().item() != 1:
                    raise ValueError("Legacy target must be one-hot")
                gold = target.argmax().item()
            domain = next((t[7:] for t in record.tags if t.startswith("domain:")), record.family)
            critical = next((t[13:] for t in record.tags if t.startswith("critical_key:")), "")
            values = {
                "states": states[0].cpu(),
                "condition_logits": condition_logits[0].cpu(),
                "parent_logits": logits[0].cpu(),
                "targets": target,
                "valid": valid,
                "gold": torch.tensor(gold, dtype=torch.long),
                "types": torch.tensor(QUESTION_TYPES.index(question.type), dtype=torch.long),
            }
            for key, value in values.items():
                collected[key].append(value)
            rows.append({
                "id": record.id, "group": record.group_id, "corpus": corpus, "split": split,
                "domain": domain, "family": record.family, "type": question.type,
                "condition": next((t[10:] for t in record.tags if t.startswith("condition:")), "none"),
                "critical_position": mapping.get(critical, -1), "tags": record.tags,
                "positions": mapping, "input_tokens": tokens,
                "modality": "image" if record.request.state.images else "text",
            })
            completed += 1
            if completed % 200 == 0 or completed == len(items):
                progress = {"encoded": completed, "total": len(items), "split": split,
                            "elapsed_seconds": time.perf_counter() - started}
                write(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        tensors = {key: torch.stack(values) for key, values in collected.items()}
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("Nonfinite features")
        folder = output / split
        folder.mkdir(parents=True)
        save_file(tensors, folder / "features.safetensors")
        write(folder / "records.json", rows)
        manifest = {
            "complete": True, "rows": len(rows), "split": split,
            "features_sha256": digest(folder / "features.safetensors"),
            "metadata_sha256": digest(folder / "records.json"),
            "specification_sha256": digest(output / "specification.json"),
        }
        write(folder / "manifest.json", manifest)
        split_manifests[split] = digest(folder / "manifest.json")
        print(json.dumps({"completed_split": split, "rows": len(rows)}), flush=True)
    write(output / "manifest.json", {"complete": True, "split_manifests": split_manifests,
          "elapsed_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
