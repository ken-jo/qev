"""Prepare a self-contained adapter release folder without contacting a publishing API."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file

from veyra.workflow_release_gate import require_release_acceptance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def percent(value):
    return "not measured" if value is None else f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--typed-report", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--photo-http-report", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--release-acceptance",
        type=Path,
        default=Path("reports/workflow-v12/release-acceptance.json"),
    )
    args = parser.parse_args()
    release_protocol = Path("configs/workflow-release-v12.json")
    acceptance = require_release_acceptance(
        args.release_acceptance, args.checkpoint, release_protocol
    )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("release folder must be empty")
    manifest = json.loads((args.checkpoint / "manifest.json").read_text())
    final = json.loads(args.final_report.read_text())
    baseline = json.loads(args.baseline_report.read_text())
    typed = json.loads(args.typed_report.read_text())
    runtime = json.loads(args.runtime_report.read_text())
    photo_http = json.loads(args.photo_http_report.read_text())
    selection = json.loads(args.selection.read_text())
    expected = digest(args.checkpoint / "head.safetensors")
    expected_manifest = digest(args.checkpoint / "manifest.json")
    if manifest["weights_sha256"] != expected:
        raise ValueError("manifest weights mismatch")
    if (
        final["protocol"]["weights_sha256"] != expected
        or typed["protocol"]["weights_sha256"] != expected
    ):
        raise ValueError("accuracy reports do not describe the packaged weights")
    if runtime["checkpoint_weights_sha256"] != expected:
        raise ValueError("runtime report describes different weights")
    if (
        photo_http["protocol"]["weights_sha256"] != expected
        or photo_http["protocol"]["manifest_sha256"] != expected_manifest
    ):
        raise ValueError("real-photo HTTP report describes a different checkpoint")
    if (
        any(result["protocol"]["manifest_sha256"] != expected_manifest for result in (final, typed))
        or runtime["checkpoint_manifest_sha256"] != expected_manifest
    ):
        raise ValueError("reports describe different calibration or runtime configuration")
    if any(
        result["protocol"]["arguments"]["split"] != "test" for result in (final, baseline, typed)
    ):
        raise ValueError("release accuracy must come from test-split evaluation")
    if final["protocol"]["records_sha256"] != baseline["protocol"]["records_sha256"]:
        raise ValueError("before/after evaluation datasets differ")
    if final["protocol"]["subset_sha256"] != baseline["protocol"]["subset_sha256"]:
        raise ValueError("before/after evaluated subsets differ")
    if selection["final_or_calibration_used"]:
        raise ValueError("final evaluation contaminated checkpoint selection")
    if not selection["selected"]["selection"][0]:
        raise ValueError("selected candidate failed development guardrails")
    if not runtime["http"]["p95_target_met"] or not photo_http["p95_target_met"]:
        raise ValueError("resident HTTP latency target was not met")
    if not all(
        runtime["protocol"][name]
        for name in (
            "choice_id_order_pass",
            "zero_generated_tokens",
            "http_path_escape_rejected",
            "http_token_budget_rejected",
        )
    ):
        raise ValueError("runtime contract checks failed")
    selected_path = Path(selection["selected"]["checkpoint"])
    if digest(selected_path / "head.safetensors") != expected:
        raise ValueError("packaged weights differ from the selected development candidate")
    if manifest["training"].get("intermediate", True) or not manifest["training"]["completed"]:
        raise ValueError("checkpoint has not completed calibration")
    tensors = load_file(args.checkpoint / "head.safetensors")
    if "readout.weight" not in tensors or not any(".lora_" in key for key in tensors):
        raise ValueError("expected readout and backbone adapters")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name in ("head.safetensors", "manifest.json"):
        shutil.copyfile(args.checkpoint / name, output / name)
    for name in ("LICENSE", "NOTICE", "ROADMAP.md"):
        shutil.copyfile(name, output / name)
    shutil.copyfile(release_protocol, output / "release-protocol.json")
    write(output / "release-readiness.json", acceptance)
    shutil.copytree("docs/data-licenses", output / "data-licenses")
    (output / "runtime").mkdir()
    shutil.copyfile(args.wheel, output / "runtime" / args.wheel.name)
    (output / "reproducibility").mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(name, output / "reproducibility" / name)
    write(
        output / "reproducibility" / "build-environment.json",
        {
            "python": sys.version,
            "wheel_sha256": digest(args.wheel),
            "prepare_source_sha256": digest(Path(__file__)),
            "installed_versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "numpy",
                    "safetensors",
                    "pillow",
                    "huggingface-hub",
                    "fastapi",
                    "uvicorn",
                    "pydantic",
                )
            },
        },
    )
    reports = {
        "foundation-final.json": args.final_report,
        "foundation-baseline.json": args.baseline_report,
        "typed-decisions.json": args.typed_report,
        "typed-baseline.json": Path("reports/foundation-v11/typed-baseline-standardized.json"),
        "typed-regression-audit.json": Path("reports/foundation-v11/typed-regression-audit.json"),
        "runtime.json": args.runtime_report,
        "photo-http.json": args.photo_http_report,
        "selection.json": args.selection,
        "objective-comparison.json": Path("runs/foundation-v11-head/results.json"),
        "laya-base.json": Path("runs/foundation-v11-laya/official-original-base/evaluation.json"),
        "laya-specialist.json": Path(
            "runs/foundation-v11-laya/official-original-specialist/evaluation.json"
        ),
        "laya-base-full-input.json": Path(
            "runs/foundation-v11-laya/official-original-base-full-input/evaluation.json"
        ),
        "laya-foundation-base.json": Path(
            "runs/foundation-v11-laya/foundation-base/evaluation.json"
        ),
        "laya-foundation-specialist.json": Path(
            "runs/foundation-v11-laya/foundation-specialist/evaluation.json"
        ),
        "laya-input-base.json": Path("runs/foundation-v11-laya/original-base-input-audit.json"),
        "laya-input-specialist.json": Path(
            "runs/foundation-v11-laya/original-specialist-input-audit.json"
        ),
        "laya-input-foundation.json": Path(
            "runs/foundation-v11-laya/foundation-base-input-audit.json"
        ),
        "laya-input-base-full.json": Path(
            "runs/foundation-v11-laya/original-base-full-input-audit.json"
        ),
        "legacy-policy-regression.json": Path(
            "runs/foundation-v11-evaluation/legacy-policy-regression/evaluation.json"
        ),
        "paired-comparison.json": Path("runs/foundation-v11-evaluation/paired-comparison.json"),
        "data-protocol.json": Path("data/foundation-v11/protocol.json"),
    }
    (output / "reports").mkdir()
    for name, path in reports.items():
        shutil.copyfile(path, output / "reports" / name)
    legacy = json.loads(reports["legacy-policy-regression.json"].read_text(encoding="utf-8"))
    if (
        legacy["protocol"]["weights_sha256"] != expected
        or legacy["protocol"]["manifest_sha256"] != expected_manifest
        or not legacy["protocol"]["arguments"]["legacy_regression"]
    ):
        raise ValueError("legacy regression does not describe the packaged checkpoint")
    if any(legacy["by_modality"][kind]["accuracy"] < 0.9 for kind in ("text", "image")):
        raise ValueError("the previous synthetic-policy 90% target was not retained")
    sample = {
        "state": {"text": "The customer was charged twice and requests a refund."},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this request?",
                "criteria": {
                    "billing": "Payments, invoices and refunds",
                    "technical": "Software faults",
                },
            }
        },
    }
    write(output / "sample_request.json", sample)
    (output / "load_veyra.py").write_text(
        '''"""Run a local request using the accompanying Veyra checkpoint."""
import argparse
import json
from pathlib import Path
from veyra.option_model import OptionModel
from veyra.schema import DecisionRequest

parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--image-root", type=Path)
parser.add_argument("--device", default="cuda")
parser.add_argument("--cache-dir", default=".cache/huggingface")
args = parser.parse_args()
model = OptionModel.load(Path(__file__).resolve().parent, device=args.device,
                         cache_dir=args.cache_dir, merge=True)
request = DecisionRequest.from_json(args.request.read_text(encoding="utf-8"))
print(json.dumps(model.predict(request, args.image_root), ensure_ascii=False, indent=2))
''',
        encoding="utf-8",
        newline="\n",
    )
    (output / "upload_to_hub.py").write_text(
        '''"""Publish this prepared folder after choosing an account/repository explicitly."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, get_token

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--public", action="store_true")
args = parser.parse_args()
folder = Path(__file__).resolve().parent
expected_files = json.loads((folder / "checksums.json").read_text())
for name, expected in expected_files.items():
    path = (folder / name).resolve()
    if not path.is_relative_to(folder) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Package checksum mismatch: " + name)
readiness = json.loads((folder / "release-readiness.json").read_text(encoding="utf-8"))
required = ("workflow_generalization", "uncertainty_probability", "uncertainty_abstention",
            "retention_development", "legacy_text", "legacy_image", "official_workflow",
            "runtime", "single_network", "provenance")
if readiness.get("release_allowed") is not True or readiness.get("required_priorities") != [1, 2]:
    raise SystemExit("Release blocked: both mandatory roadmap priorities must pass.")
if any(readiness.get("checks", {}).get(key) is not True for key in required):
    raise SystemExit("Release blocked: incomplete acceptance evidence.")
for name, key in (("head.safetensors", "weights_sha256"), ("manifest.json", "manifest_sha256"),
                  ("release-protocol.json", "release_protocol_sha256")):
    if hashlib.sha256((folder / name).read_bytes()).hexdigest() != readiness.get(key):
        raise SystemExit("Release blocked: acceptance fingerprint mismatch: " + name)
if not get_token():
    raise SystemExit("Log in with `hf auth login` first; do not put a token in a source file.")
api = HfApi()
api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=folder,
                  allow_patterns=[*expected_files, "checksums.json"],
                  ignore_patterns=[".cache/**", "__pycache__/**", ".git/**"],
                  commit_message="Publish evaluated Veyra research checkpoint")
print("https://huggingface.co/" + args.repo_id)
''',
        encoding="utf-8",
        newline="\n",
    )
    table = []
    for domain, label in (
        ("image_waste", "TrashNet real photographs"),
        ("image_leaf", "Beans photographs (legacy regression)"),
        ("text_nli", "SNLI text entailment and dynamic policies"),
        ("text_intent", "BANKING77 sampled 8-way decisions and policies"),
        ("retention", "Controlled policies (legacy regression)"),
    ):
        old, new = baseline["by_domain"][domain], final["by_domain"][domain]
        table.append(
            f"| {label} | {new['labeled_questions']} | {percent(old['accuracy'])} "
            f"| {percent(new['accuracy'])} |"
        )
    base_ref = json.loads(reports["laya-base.json"].read_text())
    typed_baseline = json.loads(reports["typed-baseline.json"].read_text())
    specialist_ref = json.loads(reports["laya-specialist.json"].read_text())
    base_foundation = json.loads(reports["laya-foundation-base.json"].read_text())
    specialist_foundation = json.loads(reports["laya-foundation-specialist.json"].read_text())
    complete_base = json.loads(reports["laya-base-full-input.json"].read_text())
    truncated_base = json.loads(reports["laya-input-base.json"].read_text())["counts"]["overall"]
    waste_views = final["by_domain_view"]
    waste_recognition = percent(waste_views["image_waste/recognition"]["accuracy"])
    waste_policy = percent(waste_views["image_waste/policy"]["accuracy"])
    waste_proposition = percent(waste_views["image_waste/proposition"]["accuracy"])
    novel_intents = final["by_intent_novelty"]["split_novel"]
    roadmap_table = "\n".join(
        (
            "| 1 | Nested workflow rules, exceptions, precedence and error costs | "
            "Better results on unseen workflow families while retaining "
            "photo and sentence/intent performance |",
            "| 2 | Probability quality and abstention with missing or conflicting evidence | "
            "Lower probability error and costly mistakes at matched answer coverage |",
            "| 3 | Longer context, OCR, visual relationships and Korean | "
            "Separate data, accuracy, input limits and latency measurements for each extension |",
        )
    )
    card = f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: veyra
language:
- en
tags:
- multimodal
- dynamic-classification
- calibrated-decisions
- research
datasets:
- stanfordnlp/snli
- PolyAI/banking77
- garythung/trashnet
- AI-Lab-Makerere/beans
---
# Veyra Foundation — dynamic decisions over text and images

A research checkpoint for request-defined `choice`, ordinal `score`, and `noul` (P(true)) decisions.
Text alone and one image with text use the same Qwen3.5-2B network. It returns distributions in a
single forward pass and generates zero answer tokens. It does not generate new schemas or labels:
callers supply candidate meanings or rubrics with each request.

## Current scope

The evaluated use cases are photograph recognition with supplied policies, sentence relations,
and decisions among request-defined intent candidates, primarily in English. Complex workflow
reasoning, longer context, OCR, visual relationships and general Korean performance are planned
improvement areas. Their current limits and measured workflow regression remain documented below.

The release uses one network and zero generated answer tokens. Its resident photo HTTP p95 is
**{photo_http["latency_ms"]["p95"]:.2f} ms** on RTX 4060 Ti 8 GB
under the workload described in Run.
The **300 ms** target applies to that reference workload; expanded inputs need separate timing.

## What is in this repository

This is a custom Veyra adapter/readout checkpoint, **not a standalone Transformers AutoModel**.
`head.safetensors` contains {sum(t.numel() for t in tensors.values()):,} stored adaptation/readout
parameters. The Apache-2.0 Qwen base is downloaded separately at pinned revision
`{manifest["backbone"]["revision"]}`. Vision weights stay frozen; the model lineage includes late
language-layer LoRA adaptation. The exact selected training path is in `manifest.json` and the
development-only selection report. Runtime requirements are pinned in the included wheel.

## Accuracy

All hard-label decisions are counted, including abstentions. Multiple question views share the
same observation group. Group bootstrap intervals and soft-target metrics are in `reports/`.

| Evaluation | Decisions | Dynamic v2 before | This checkpoint |
| --- | ---: | ---: | ---: |
{chr(10).join(table)}

TrashNet's separate views, 407 questions each: recognition **{waste_recognition}**,
request-defined policy **{waste_policy}**, proposition **{waste_proposition}**.

TrashNet is photographed waste, often on a plain background; it is not a general real-world vision
benchmark. Three questions per observation measure recognition, a supplied arbitrary priority
mapping, and a proposition. Those priorities are not real recycling regulations or disease severity.
BANKING77 uses sampled eight-candidate subsets, not the published 77-way benchmark. Nine intent
classes are held out from adaptation candidate pools. The 924-question intent aggregate includes
all 77 intents: 60 trained intents, eight development-only intents and nine final-only intents.
The nine final-only intents contribute {novel_intents["labeled_questions"]} questions and score
**{percent(novel_intents["accuracy"])}**. Dataset groups are split before training. Exact duplicates
and near duplicates found by the declared pHash/dHash/color thresholds are grouped.
Object identity and Qwen pretraining overlap are unknown.
The Beans baseline is measured one question per call here (78.65%); the older three-question
batched release evaluation reported 79.17%. The before/after table uses matched calls.

On the previous 3,576-question synthetic-policy final set, this checkpoint scores
**{percent(legacy["by_modality"]["text"]["accuracy"])}** for text and
**{percent(legacy["by_modality"]["image"]["accuracy"])}** for image policies.
Dynamic v2 scored 93.80% and 97.71% respectively. This is a legacy regression check;
the set was excluded from the new training, selection and calibration.

On the separate official typed-decisions test (2,000 teacher-labeled questions), measured accuracy:
Veyra **{percent(typed["overall"]["accuracy"])}**,
LAYA base **{percent(base_ref["overall"]["accuracy"])}**,
LAYA typed-decisions specialist **{percent(specialist_ref["overall"]["accuracy"])}**.
The previous Dynamic v2 checkpoint scored
**{percent(typed_baseline["overall"]["accuracy"])}** on this same benchmark.
Foundation therefore regresses on workflow selection accuracy despite improving the adapted
photo and sentence/intent domains. It is not a universal replacement for the earlier checkpoint.
The paired historical-prediction audit is in `reports/typed-regression-audit.json`.
The stock base context limit truncates state text on {truncated_base["state_truncated"]} questions;
With only `max_len=1024` changed, base accuracy is
**{percent(complete_base["overall"]["accuracy"])}**.
The specialist and the foundation text corpus fit their released input limits.
The LAYA specialist was trained on that benchmark. The official test is a previously inspected
regression benchmark for Veyra, and was excluded from this checkpoint's training and selection.
This does not establish JEV parity. LAYA source revisions, SDK version and input details accompany
the reports; model-card claims are not substituted for local measurements.

On the shared foundation text subset (2,141 questions, 2,003 with hard gold labels):

| Model | Accuracy |
| --- | ---: |
| LAYA base | {percent(base_foundation["overall"]["accuracy"])} |
| LAYA workflow specialist | {percent(specialist_foundation["overall"]["accuracy"])} |
| Veyra Foundation | {percent(final["by_modality"]["text"]["accuracy"])} |

This subset combines SNLI, sampled BANKING77 questions, controlled policies and partial evidence.
Veyra was adapted on these task families; LAYA was not. Conversely, the LAYA specialist was
adapted on the official workflow benchmark. These results do not identify a universally best model.

## Run

Use Python 3.12 and a CUDA-capable PyTorch installation. The exercised machine is Windows with an
RTX 4060 Ti 8 GB. Install the included wheel, then run the accompanying loader:

```sh
pip install runtime/{args.wheel.name} --extra-index-url https://download.pytorch.org/whl/cu128
python load_veyra.py --request sample_request.json
```

For a resident HTTP service (avoids model loading per request):

```sh
veyra serve --checkpoint . --image-root /absolute/path/to/images
```

It binds to `127.0.0.1:8000`; POST Veyra requests to `/v1/systemone`.
This API is not a drop-in Jev SDK.
Image paths are relative to `--image-root`. One image, 1–4 questions, 2–16 choice/score candidates
and at most 2,048 total processed tokens are supported. Longer inputs are rejected explicitly.
The detailed measured latency matrix and actual HTTP measurements are in `reports/runtime.json`;
cold loading, concurrent requests and large images have separate costs.
For 40 serial requests with distinct held-out TrashNet photographs, one recognition question and
six candidates, resident local HTTP p95 is **{photo_http["latency_ms"]["p95"]:.2f} ms**
on the RTX 4060 Ti.
That measurement includes image decode and the HTTP/JSON round trip, uses three warmups, and
excludes model loading. Full image sizes and input-token counts are in `reports/photo-http.json`.

## Training and probability interpretation

Five objectives were compared with three declared seeds each on identical training representations:
cross-entropy, proper scoring loss, its noisy differentiable form, matched RLOO, and probability
transport across recognition/policy views. This five-way comparison updated the readouts on
frozen Dynamic v2 representations. The selected common backbone was then considered for
further LoRA adaptation. Selection uses development groups; temperatures and abstention thresholds
use separate calibration groups; final groups are evaluated after selection.

Proper scoring losses and post-training temperature fitting do not guarantee calibrated confidence
under distribution shift. `abstained` reflects an empirical calibration-set policy, not a guaranteed
risk bound. Known synthetic partial-evidence targets are exact conditional probabilities; older
missing-evidence annotations are not empirical frequencies. English is the primary evaluated
language. General Korean accuracy, OCR, fine visual relationships and production decision safety
are unverified. No unrestricted general-purpose 90% accuracy claim is made.

## Roadmap

Complex decision improvements are recorded as future research milestones:

| Priority | Planned improvement | Required evidence |
| --- | --- | --- |
{roadmap_table}

[Full roadmap](ROADMAP.md) · [한국어 로드맵](ROADMAP.ko.md).
Priorities 1 and 2 are mandatory release requirements under the frozen
`release-protocol.json`. Publication requires every check in `release-readiness.json` to pass
for these exact weights and calibration. The historical Foundation measurements describe the
baseline; they do not establish completion of the new requirements. Priority 3 remains future work.

## Sources and licenses

Code and Veyra trained components: Apache-2.0. Qwen base: Apache-2.0. TrashNet and Beans: MIT;
SNLI: CC-BY-SA-4.0; BANKING77: CC-BY-4.0. Original dataset bytes are not redistributed here.
Attributions and full license references are in `NOTICE`, `data-licenses/` and the data protocol.
Kaggle `vminhkhoi/trashnet` version 1 was matched by class and file bytes
to original-author TrashNet.

Project: https://github.com/ken-jo/veyra
References: https://huggingface.co/convaiinnovations/laya and
https://huggingface.co/convaiinnovations/laya-typed-decisions
"""
    (output / "README.md").write_text(card, encoding="utf-8", newline="\n")
    checksums = {
        str(p.relative_to(output)).replace("\\", "/"): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write(output / "checksums.json", checksums)
    write(
        output.parent / (output.name + "-preparation.json"),
        {
            "prepared": True,
            "published": False,
            "folder": str(output.resolve()),
            "weights_sha256": expected,
            "files": len(checksums) + 1,
            "checksums_sha256": digest(output / "checksums.json"),
            "auth_required_for_publish": True,
        },
    )
    print(
        json.dumps(
            {"prepared": str(output.resolve()), "files": len(checksums) + 1, "published": False}
        )
    )


if __name__ == "__main__":
    main()
