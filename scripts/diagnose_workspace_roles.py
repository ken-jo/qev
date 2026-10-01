"""Measure whether four supervised positions carry distinct condition information."""

import gc
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from train_foundation_head import write
from workflow_workspace_supervision import WorkspaceSupervisor

from veyra.data import TrainingRecord
from veyra.option_model import OptionModel
from veyra.workflow_facts import stated_probabilities


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(probability, hidden, target, training_prior):
    observed = (target == 0) | (target == 1)
    broadcast = probability.mean(-1, keepdim=True).expand_as(probability)
    normalized = F.normalize(F.layer_norm(hidden.float(), (hidden.shape[-1],)), dim=-1)
    cosine = normalized @ normalized.transpose(-1, -2)
    off_diagonal = ~torch.eye(4, dtype=torch.bool)

    def observed_accuracy(values, mask):
        return float(((values >= 0.5)[mask] == (target == 1)[mask]).float().mean())

    mixed = (target.max(-1).values - target.min(-1).values) > 1e-6
    permutations = list(itertools.permutations(range(4)))
    permutation_error = torch.stack(
        [(probability[:, list(order)] - target).square().mean(-1) for order in permutations], -1
    )
    prior = training_prior[None].expand_as(target)
    return {
        "questions": len(target),
        "observed_conditions": int(observed.sum()),
        "mixed_condition_questions": int(mixed.sum()),
        "condition_probability_mse": float((probability - target).square().mean()),
        "broadcast_same_request_prediction_mse": float((broadcast - target).square().mean()),
        "training_condition_prior_mse": float((prior - target).square().mean()),
        "observed_condition_accuracy": observed_accuracy(probability, observed),
        "broadcast_observed_accuracy": observed_accuracy(broadcast, observed),
        "training_prior_observed_accuracy": observed_accuracy(prior, observed),
        "per_condition_observed_accuracy": {
            name: observed_accuracy(probability, observed & (torch.arange(4) == i))
            for i, name in enumerate("ABCD")
        },
        "mean_within_request_probability_std": float(probability.std(-1, unbiased=False).mean()),
        "mean_pairwise_normalized_hidden_cosine": float(cosine[:, off_diagonal].mean()),
        "fraction_requests_with_identical_four_binary_predictions": float(
            ((probability >= 0.5).sum(-1).remainder(4) == 0).float().mean()
        ),
        "mixed_question_probability_mse": float(
            (probability[mixed] - target[mixed]).square().mean()
        ),
        "mixed_question_broadcast_mse": float((broadcast[mixed] - target[mixed]).square().mean()),
        "mean_mse_over_all_24_role_permutations": float(permutation_error.mean()),
        "best_assignment_per_question_mse_reference": float(
            permutation_error.min(-1).values.mean()
        ),
        "target_mean_within_request_std": float(target.std(-1, unbiased=False).mean()),
    }


def main():
    selection_path = Path("runs/workflow-v12-workspace-selection/selection.json")
    evaluation = read("runs/workflow-v12-workspace-evaluation/progress.json")
    selection = read(selection_path)
    if not (
        selection["eligible"] is False
        and evaluation["stage"] == "selection"
        and evaluation["status"] == "gate_failed"
        and evaluation["final_predictions_started"] is False
    ):
        raise ValueError(
            "run only after the existing GPU evaluation has stopped at its development gate"
        )
    output = Path("runs/workflow-v12-workspace-role-diagnostic")
    if output.exists():
        raise FileExistsError("role diagnosis is immutable")
    config_path = Path("configs/workflow-workspace-study-v12.json")
    config = read(config_path)
    if selection["study_protocol_sha256"] != digest(config_path):
        raise ValueError("selection used another study")
    records_path = Path(config["records"])
    if digest(records_path) != config["records_sha256"]:
        raise ValueError("original records changed")
    records = {"train": [], "dev": []}
    for line in records_path.open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] not in records:
            continue
        if {"domain:workflow_new", "domain:uncertainty"}.intersection(raw["tags"]):
            records[raw["split"]].append(TrainingRecord.model_validate(raw))
    if len(records["train"]) != 3840 or len(records["dev"]) != 1200:
        raise ValueError("expected original eligible condition-supervision population")
    target = torch.tensor([stated_probabilities(r.request.state.text) for r in records["dev"]])
    prior = torch.tensor(
        [stated_probabilities(r.request.state.text) for r in records["train"]]
    ).mean(0)
    decoder_path = Path(
        "runs/workflow-v12-workspace-study/primitive/epoch-2/auxiliary-training-only.safetensors"
    )
    candidates = [selection["candidates"][index] for index in (0, 2, 4)]
    for candidate in candidates:
        checkpoint = Path(candidate["checkpoint"])
        for name, key in (
            ("head.safetensors", "weights_sha256"),
            ("manifest.json", "manifest_sha256"),
        ):
            if digest(checkpoint / name) != candidate[key]:
                raise ValueError("declared checkpoint changed")
    protocol = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": digest(selection_path),
        "study_sha256": digest(config_path),
        "records_sha256": digest(records_path),
        "source_sha256": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("scripts/workflow_workspace_supervision.py"),
                Path("src/veyra/workflow_facts.py"),
                Path("src/veyra/option_model.py"),
            )
        },
        "candidates": [
            {key: item[key] for key in ("checkpoint", "weights_sha256", "manifest_sha256")}
            for item in candidates
        ],
        "fixed_decoder": str(decoder_path),
        "fixed_decoder_sha256": digest(decoder_path),
        "development_ids": [r.id for r in records["dev"]],
        "training_condition_prior": prior.tolist(),
        "hypothesis": (
            "The four identical dot tokens may have weak condition-role binding, explaining "
            "small supervised-versus-control gains. Compare each position's decoded probability "
            "with broadcasting their request mean, training priors and role permutations."
        ),
        "model_training_or_policy_fitting": False,
        "calibration_or_final_used": False,
        "merged_bf16": True,
        "scope": "Exploratory development diagnosis after failed selection; no release decision.",
    }
    write(output / "protocol.json", protocol)
    torch.set_num_threads(4)
    results = []
    for index, candidate in enumerate(candidates):
        model = OptionModel.load(Path(candidate["checkpoint"]), local_files_only=True, merge=True)
        model.eval()
        supervisor = WorkspaceSupervisor(model, config["seed"])
        supervisor.decoder.load_state_dict(load_file(decoder_path))
        probabilities, states = [], []
        with torch.inference_mode():
            for count, record in enumerate(records["dev"], 1):
                with supervisor.observe(True):
                    model(record.request, records_path.parent)
                    probabilities.append(supervisor.logits()[0].sigmoid().cpu())
                    states.append(supervisor.captured[0].cpu())
                if count % 300 == 0:
                    progress = {"candidate": index, "evaluated": count, "total": len(target)}
                    write(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        probability, hidden = torch.stack(probabilities), torch.stack(states)
        report = summarize(probability, hidden, target, prior)
        artifact = output / f"candidate-{index}.safetensors"
        save_file({"probability": probability, "hidden": hidden, "target": target}, artifact)
        results.append(
            {
                "checkpoint": candidate["checkpoint"],
                "artifact_sha256": digest(artifact),
                "metrics": report,
            }
        )
        write(output / "results.json", results)
        supervisor.close()
        del supervisor, model, hidden, probability, probabilities, states
        gc.collect()
        torch.cuda.empty_cache()
    write(
        output / "complete.json",
        {
            "completed": True,
            "protocol_sha256": digest(output / "protocol.json"),
            "results_sha256": digest(output / "results.json"),
            "release_allowed": False,
            "limitations": (
                "The decoder was trained only with the primitive arm, so transferring it to "
                "other checkpoints is not a capacity comparison with independently refitted "
                "probes. Similar neighboring hidden states alone do not prove collapse; compare "
                "decoded role information against the request-mean baseline. Best-assignment "
                "MSE uses targets as a diagnostic reference and is not model performance. "
                "No inference prompt, model weights, calibration or release gate was changed."
            ),
        },
    )
    print(json.dumps({"completed": True, "release_allowed": False}), flush=True)


if __name__ == "__main__":
    main()
