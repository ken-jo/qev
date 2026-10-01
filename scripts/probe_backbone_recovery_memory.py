"""Measure training memory on the longest reserved training inputs only."""

import gc
from pathlib import Path

import torch
from study_readout_recovery import cache, read
from train_foundation_head import write
from train_workflow_backbone import forward
from workflow_depth_common import layer_number, training_mode

from veyra.data import read_records
from veyra.option_model import OptionModel


def main():
    config = read("configs/workflow-backbone-recovery-v13.json")
    tensors, rows = cache("train")
    output = Path("runs/backbone-recovery-memory-probe-v13.json")
    if output.exists():
        raise FileExistsError("memory probe is immutable")
    records, roots = {}, {}
    for corpus, spec in config["source_datasets"].items():
        path = Path(spec["path"])
        for record in read_records(path):
            if record.split == "train":
                records[(corpus, record.id)] = record
        roots[corpus] = path.parent
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.cuda.set_per_process_memory_fraction(0.90)
    model = OptionModel.load(config["parent"], local_files_only=True, merge=False)
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if ".lora_" in name and layer_number(name) in config["trainable_adapter_layers"]:
            parameter.requires_grad_(True)
    model.readout.requires_grad_(True)
    model.binding_head.requires_grad_(True)
    model.encoder.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    for index, layer in enumerate(model.encoder.model.language_model.layers):
        layer.gradient_checkpointing = index in config["trainable_adapter_layers"]
    training_mode(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0)
    scales = torch.tensor(model.calibration.temperatures, device=model.encoder.device)

    def step(index):
        row = rows[index]
        logits, states = forward(model, records[(row["corpus"], row["id"])], roots[row["corpus"]])
        valid = tensors["valid"][index : index + 1].to(logits.device)
        target = tensors["targets"][index : index + 1].to(logits.device)
        logp = (logits / scales[tensors["types"][index]]).masked_fill(~valid, -1e9).log_softmax(-1)
        loss = -(target * logp).sum()
        if row["corpus"] == "workflow":
            teacher = tensors["parent_logits"][index : index + 1].to(logits.device) / 2
            teacher = teacher.masked_fill(~valid, -1e9).softmax(-1)
            student = (logits / 2).masked_fill(~valid, -1e9).log_softmax(-1)
            loss = loss + 4 * (teacher * (teacher.clamp_min(1e-12).log() - student)).sum()
        (loss / 8).backward()

    indices = []
    for modality in ("text", "image"):
        indices.extend(
            sorted(
                [i for i, row in enumerate(rows) if row["modality"] == modality],
                key=lambda i: (-rows[i]["input_tokens"], rows[i]["id"]),
            )[:8]
        )
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    for count, index in enumerate(indices, start=1):
        step(index)
        if count % 8 == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        print(
            {
                "completed": count,
                "tokens": rows[index]["input_tokens"],
                "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
            },
            flush=True,
        )
    write(
        output,
        {
            "passed": True,
            "training_questions": len(indices),
            "maximum_input_tokens": max(rows[i]["input_tokens"] for i in indices),
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "per_process_memory_fraction": 0.9,
            "parameter_updates": False,
            "development_calibration_or_final_used": False,
            "purpose": (
                "Training-only allocation feasibility; not long-run memory or accuracy proof"
            ),
            "records": [rows[i]["id"] for i in indices],
            "release_allowed": False,
        },
    )


if __name__ == "__main__":
    main()
