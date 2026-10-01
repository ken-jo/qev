"""Build a workflow checkpoint package only after both mandatory priorities pass."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

from safetensors.torch import load_file

from veyra.workflow_release_gate import require_release_acceptance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def percent(value):
    return "not measured" if value is None else f"{100 * value:.2f}%"


def architecture_note(manifest, parameters):
    adaptation = manifest["adaptation"]
    note = (
        f"The stored adaptation/readout has {parameters:,} parameters. Qwen revision is\n"
        f"`{manifest['backbone']['revision']}`. Vision weights are frozen. "
        f"Rank-{adaptation['rank']} LoRA covers {adaptation['layers']} language layers; "
        "the decision readouts were also adapted."
    )
    training = manifest.get("training", {})
    method = training.get("method")
    if method in {"depth-full_depth", "depth-late_control"}:
        arm = method.removeprefix("depth-")
        note += (
            f"\nSelected training checkpoint: `{arm}`, epoch {training['epochs']}, "
            f"seed {training['seed']}. "
        )
        note += (
            "This arm enabled learning in both earlier and later language layers."
            if method == "depth-full_depth"
            else "The earlier 12 layers' adapters had zero learning rates; training continued "
            "in the later 12 layers' adapters and decision readouts."
        )
    return note


def portable_roadmap(content):
    """Keep repository-only report/configuration links usable in the HF package."""

    def replace(match):
        target = match.group(2)
        if target.startswith(("configs/", "reports/")):
            path = Path(target.split("#", 1)[0])
            if not path.is_file():
                raise ValueError("roadmap references a missing repository artifact: " + target)
            target = "https://github.com/ken-jo/veyra/blob/main/" + target
        return match.group(1) + target + ")"

    return re.sub(r"(\[[^\]]+\]\()([^)]+)\)", replace, content)


def evidence_packaging_plan(evidence, root):
    """Verify every local input, then keep original data outside the distributable bundle."""
    metadata_names = {
        "audit.json",
        "independent-target-audit.json",
        "manifest.json",
        "protocol.json",
        "specification.json",
    }
    plan = {"bundled": {}, "external_inputs": {}}
    root = Path(root).resolve()
    for source, expected in evidence.items():
        path = Path(source).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("invalid or changed acceptance evidence: " + source)
        relative = path.relative_to(root)
        entry = {"repository_path": relative.as_posix(), "sha256": expected}
        raw_input = relative.parts[0] in {"data", "datasets"} and path.name not in metadata_names
        if raw_input:
            plan["external_inputs"][source] = entry
        else:
            entry["package_path"] = (Path("evidence") / relative).as_posix()
            plan["bundled"][source] = entry
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, default=Path("runs/workflow-v12-evaluation"))
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    report_path = args.evaluation / "release-acceptance.json"
    protocol_path = Path("configs/workflow-release-v12.json")
    acceptance = require_release_acceptance(report_path, args.checkpoint, protocol_path)
    evidence_plan = evidence_packaging_plan(acceptance["evidence_files"], root)
    if args.output.exists():
        raise FileExistsError("prepared package is immutable")
    # A locally built wheel must contain the exact implementation used for evaluation.
    with zipfile.ZipFile(args.wheel) as archive:
        for path in sorted(Path("src/veyra").glob("*.py")):
            if archive.read("veyra/" + path.name) != path.read_bytes():
                raise ValueError("runtime wheel does not match the evaluated source: " + str(path))
    final = read(args.evaluation / "final/evaluation.json")
    baseline = read(args.evaluation / "baseline/evaluation.json")
    paired = read(args.evaluation / "paired.json")
    official = read(args.evaluation / "official-regression/evaluation.json")
    legacy = read(args.evaluation / "legacy-regression/evaluation.json")
    regression = read(args.evaluation / "foundation-regression/evaluation.json")
    manifest = read(args.checkpoint / "manifest.json")
    output = args.output
    output.mkdir(parents=True)
    for name in ("head.safetensors", "manifest.json"):
        shutil.copyfile(args.checkpoint / name, output / name)
    for name in ("LICENSE", "NOTICE", "ROADMAP.md"):
        shutil.copyfile(name, output / name)
    for name in ("ROADMAP.md"):
        (output / name).write_text(
            portable_roadmap(Path(name).read_text(encoding="utf-8")),
            encoding="utf-8",
            newline="\n",
        )
    shutil.copyfile(protocol_path, output / "release-protocol.json")
    shutil.copytree("docs/data-licenses", output / "data-licenses")
    (output / "runtime").mkdir()
    shutil.copyfile(args.wheel, output / "runtime" / args.wheel.name)
    relocation = {}
    for entry in evidence_plan["bundled"].values():
        path = root / entry["repository_path"]
        relative = Path(entry["package_path"])
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        relocation[relative.as_posix()] = entry["sha256"]
    original = output / "evidence/original-acceptance.json"
    shutil.copyfile(report_path, original)
    historical_reports = (
        "foundation-final.json",
        "typed-regression.json",
        "typed-baseline-standardized.json",
        "laya-official-original-base.json",
        "laya-official-original-specialist.json",
        "laya-original-base-input-audit.json",
        "laya-sources.json",
    )
    historical_path = output / "evidence/historical-foundation"
    historical_path.mkdir()
    for name in historical_reports:
        shutil.copyfile(Path("reports/foundation-v11") / name, historical_path / name)
    readiness = {
        **acceptance,
        "evidence_files": relocation,
        "original_acceptance_report": "evidence/original-acceptance.json",
        "original_acceptance_sha256": digest(report_path),
        "evidence_relocated_for_portable_package": True,
        "evidence_mapping": {
            source: entry["package_path"] for source, entry in evidence_plan["bundled"].items()
        },
        "external_input_files": evidence_plan["external_inputs"],
        "original_dataset_inputs_bundled": False,
    }
    write(output / "release-readiness.json", readiness)
    # Reuse the stable sample and loader bytes without executing the historical packager.
    historical = ast.parse(Path("scripts/prepare_huggingface.py").read_text(encoding="utf-8"))
    loader = None
    for node in ast.walk(historical):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                node.func.attr == "write_text"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                value = node.args[0].value
                if isinstance(value, str) and value.startswith('"""Run a local request'):
                    loader = value
    if loader is None:
        raise ValueError("stable loader source is missing")
    (output / "load_veyra.py").write_text(loader, encoding="utf-8", newline="\n")
    write(
        output / "sample_request.json",
        {
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
        },
    )
    (output / "upload_to_hub.py").write_text(UPLOAD_HELPER, encoding="utf-8", newline="\n")
    (output / "reproducibility").mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(name, output / "reproducibility" / name)
    write(
        output / "reproducibility/build-environment.json",
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
    fresh_rows = []
    for domain, label in (
        ("workflow_new", "Previously unseen procedural workflow families"),
        ("photo_guard", "CIFAR-10 low-resolution photograph guard"),
        ("text_nli", "Fresh SNLI observation groups"),
        ("text_intent", "Fresh BANKING77 sampled eight-candidate questions"),
    ):
        old, new = baseline["by_domain"][domain], final["by_domain"][domain]
        fresh_rows.append(
            f"| {label} | {new['labeled_questions']} | {percent(old['accuracy'])} "
            f"| {percent(new['accuracy'])} |"
        )
    family_rows = [
        f"| {family} | {row['groups']} | {percent(row['before'])} | {percent(row['after'])} |"
        for family, row in paired["new_workflow"]["families"].items()
    ]
    old_regression = read(historical_path / "foundation-final.json")
    if (
        old_regression["protocol"]["records_sha256"] != regression["protocol"]["records_sha256"]
        or old_regression["protocol"]["subset_sha256"] != regression["protocol"]["subset_sha256"]
        or old_regression["protocol"]["weights_sha256"]
        != read(protocol_path)["baseline_weights_sha256"]
    ):
        raise ValueError("historical Foundation comparison does not match the regression corpus")
    regression_rows = [
        f"| {domain} | {row['labeled_questions']} | "
        f"{percent(old_regression['by_domain'][domain]['accuracy'])} "
        f"| {percent(row['accuracy'])} |"
        for domain, row in regression["by_domain"].items()
        if row["accuracy"] is not None
    ]
    u = paired["uncertainty"]
    uncertain_rows = [
        f"| {label} | {u[key]['before']:.6f} | {u[key]['after']:.6f} |"
        for key, label in (
            ("nll", "NLL"),
            ("brier", "Squared distribution error"),
            ("matched_coverage_cost", "Expected error cost at 80% answer coverage"),
        )
    ]
    policy = final["risk"]["actual_policy"]
    uncertain_policy = final["detailed_by_domain"]["uncertainty"]["actual_policy"]
    parameters = sum(t.numel() for t in load_file(args.checkpoint / "head.safetensors").values())
    workflow_ci = paired["new_workflow"]["accuracy"]["delta_95_ci"]
    continuation_note = ""
    extensions = acceptance.get("release_extensions", {})
    if set(extensions) & {
        "workflow_workspace_v12",
        "workflow_curriculum_v12",
        "workflow_depth_v12",
        "workflow_depth_policy_v12",
        "workflow_cohort_policy_v12",
    }:
        prior_policy = acceptance["details"]["previous_calibration_policy_regression"]
        policy_rows = [
            f"| {kind} | {percent(value['coverage'])} | {percent(value['expected_error'])} |"
            for kind, value in prior_policy.items()
        ]
        if set(extensions) & {
            "workflow_depth_v12",
            "workflow_depth_policy_v12",
            "workflow_cohort_policy_v12",
        }:
            study_description = """
Earlier workflow, condition-supervision and prerequisite-curriculum candidates failed their
fixed calibration or development requirements before final inference. A later study compared
learning in all 24 language-layer adapters with a control whose earlier 12 adapters had zero
learning rates. Both arms used the same initialization, exact original/prerequisite queries,
order, forward/backward count, gradient clipping and AdamW groups/state. The control still
computed earlier-layer gradients; saved earlier adapter values were required to stay unchanged.
Each arm ran two epochs with one training seed. Parent and all four checkpoints entered the
same merged development selection; prerequisite diagnostics did not replace its criteria.
Training-only activation checkpointing and the added adapters introduce no inference module,
prompt change or generated answer token: all adapters merge into the original backbone matrices.
This design alone does not establish a depth advantage or robustness across seeds. Independent
final, retention, calibrated abstention and runtime evidence remain mandatory. Candidate metrics
and selection must reproduce from saved raw outputs before fresh calibration proceeds.
"""
        elif "workflow_curriculum_v12" in extensions:
            study_description = """
The first workflow candidate failed score/noul calibration, and the following matched condition-
supervision study failed the score development screen, both before final inference. A separately
declared study then compared direct prerequisite questions against extra outcome replay. Both
arms used the same parent, one seed, two epochs, original training/replay data, input-state order,
forward/backward counts and optimizer schedule. The additional questions asked individual fact
or rule truth, or repeated outcome decisions on exactly the same states. Question wording,
candidate counts and types differ, so token/FLOP budgets are not identical. No inference module
or generated answer token was added. Derived queries preserve their source groups and splits;
skill-query results were diagnostic only and never replaced original selection requirements.
The selected arm and epoch are identified in the hash-bound selection evidence. This does not
establish a prerequisite-curriculum advantage unless the matched comparison supports it, nor
does a single training seed establish robustness. All candidate development metrics and their
selection were independently recomputed from saved raw outputs before final evaluation.
"""
        else:
            study_description = """
The first workflow candidate failed score/noul calibration before any final inference. A separate
study then compared two additional epochs of direct condition supervision with a matched
continuation control, using one declared seed. Training-only condition gradients reached the
backbone at four existing internal positions. The auxiliary decoder is excluded from inference.
The selected checkpoint is identified by the hash-bound development selection in `evidence/`;
this comparison does not establish robustness across training seeds.
"""
        continuation_note = (
            "\n## Separate continuation study and calibration provenance\n"
            + study_description
            + """
Fresh calibration preserves all original training, development and final records. Its 2,644
questions use fresh observation groups. All 1,200 original public typed-decisions training cases
were already used in earlier partitions, so 600 fresh calibration questions use newly authored,
explicit policies in the same four familiar workflow families. Their targets are computed from
the stated rules, not new annotations from the original public teacher. Those two calibration
populations are not interchangeable performance benchmarks. Independent rule parsing verified
all 2,040 new procedural calibration targets; the remaining 604 targets retain public source labels.

After fitting on fresh groups, the deployed policy was applied unchanged to the exact original
policy-fitting groups from the failed experiment. Each type was required to retain at least 60%
of answers with at most 15% empirical error. These are previously inspected regression results:

| Type | Answer coverage | Expected error |
| --- | ---: | ---: |
"""
            + "\n".join(policy_rows)
            + "\n"
        )
        if "workflow_depth_policy_v12" in extensions:
            validation_policy = acceptance["details"]["independent_policy_validation"]
            validation_rows = [
                f"| {kind} | {percent(value['coverage'])} | {percent(value['expected_error'])} |"
                for kind, value in validation_policy.items()
            ]
            continuation_note += (
                """
The first depth-model calibration passed its fitting population but failed the previously
inspected score regression (16.95% expected error). A separate policy study retained exactly
the same development-selected weights. A 12% empirical fitting target was declared after
diagnosing that failure and before generating two further disjoint calibration populations.
This is a post-hoc motivated improvement, not a prospectively selected hyperparameter in the
original depth experiment. No threshold from the diagnosis was promoted directly.

Temperatures and thresholds were fitted on separate group partitions of the new fitting set.
The calibrated policy and inference/audit sources were then frozen before any prediction on
the independent 2,644-question validation set. Every type had to achieve at least 60% answer
coverage with at most 15% expected error on that entire validation population, followed by
the unchanged original-group regression above. No parameter was refitted on validation data.
The two new populations retain the same authored-policy/public-label limitations described
above. The margin and descriptive group-bootstrap intervals provide no distribution-free
guarantee of future error. Final evaluation and every original release gate remain mandatory.

Independent validation of the frozen policy:

| Type | Answer coverage | Expected error |
| --- | ---: | ---: |
"""
                + "\n".join(validation_rows)
                + "\n"
            )
        if "workflow_cohort_policy_v12" in extensions:
            continuation_note = continuation_note.replace(
                "Fresh calibration preserves all original training, development and final records. "
                "Its 2,644\n"
                "questions use fresh observation groups.",
                "The original calibration populations preserved all training, "
                "development and final\n"
                "records. Each contained 2,644 questions in fresh observation groups.",
            )
            validation_rows = [
                f"| {population} | {kind} | {percent(value['coverage'])} | "
                f"{percent(value['expected_error'])} |"
                for population, types in acceptance["details"][
                    "independent_policy_validation"
                ].items()
                for kind, value in types.items()
            ]
            continuation_note += (
                """
The first depth calibration failed previous-group score regression (16.95% error). A follow-up
with a 12% fitting target passed its fitting gates but failed noul on independent validation
(17.09%). Those failures were preserved before any final prediction. The current adaptive
cohort study explicitly reused the three inspected populations, including that failed validation,
as **fitting data**: 7,932 questions / 3,252 groups. They are not independent evidence for this
policy. The development-selected model weights were retained exactly.

Temperatures pooled only one group partition. A single threshold per type was then chosen on
the other partition from a fixed 0.005 grid. Every fitting cohort separately had to meet at most
12% expected error and at least 60% coverage. Cohort identity is used only in offline fitting;
inference still uses one existing type-specific threshold, one backbone pass and no new module.

Two new validation populations (seeds 233 and 239; 2,644 questions / 1,084 groups each) were
declared before generation and fitting. Targets and split separation were independently audited.
The policy and inference/audit sources were frozen before either validation prediction. Every
type had to meet the original 15% error / 60% coverage requirements in **both** populations,
followed by the original-group regression above. Nothing was refitted on these two validations.
The datasets' authored-rule and public-source limitations remain. Different generated samples
do not establish coverage of arbitrary deployment domains. This empirical procedure and its
descriptive group-bootstrap intervals provide no distribution-free guarantee of future risk.

Independent validation of the frozen cohort-fitted policy:

| Population | Type | Answer coverage | Expected error |
| --- | --- | ---: | ---: |
"""
                + "\n".join(validation_rows)
                + "\n"
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
- research
datasets:
- LocalLLaMA/typed-decisions
- stanfordnlp/snli
- PolyAI/banking77
- garythung/trashnet
- AI-Lab-Makerere/beans
---
# Veyra Workflow — fast dynamic decisions with explicit abstention

One Qwen3.5-2B network accepts text alone or one photograph with text and request-defined
`choice`, ordinal `score`, or `noul` (P(true)) criteria. Candidates and meanings come from the
request. The model returns probabilities and `abstained` in one backbone forward, with zero
generated answer tokens. It does not generate candidate schemas.

**This is a research model.** Both mandatory roadmap priorities passed the frozen acceptance
protocol for these exact weights. The new workflow evidence is controlled procedural transfer;
it does not establish reliable operation on arbitrary business workflows or JEV parity.

## Fresh held-out results

All hard decisions are counted, including abstentions. Before is the frozen Foundation release.
Related question views share an observation group; 2,000 paired group-bootstrap resamples are
used for the reported intervals, without multiple-comparison adjustment.

| Evaluation | Questions | Foundation | Workflow |
| --- | ---: | ---: | ---: |
{chr(10).join(fresh_rows)}

The unseen workflow accuracy improvement interval is
**[{100 * workflow_ci[0]:.2f}, {100 * workflow_ci[1]:.2f}] percentage points**.
Final families were absent from training, development and calibration.

| New family | Observation groups | Foundation | Workflow |
| --- | ---: | ---: | ---: |
{chr(10).join(family_rows)}

CIFAR-10 is an additional 32×32 image guard, not a high-resolution image benchmark. Its upstream
dataset card marks the license unknown; original images are not included. Fresh SNLI and BANKING77
groups exclude earlier selected observations, but public pretraining overlap is unknown. The
eight-candidate intent setup is not the standard BANKING77 77-way benchmark.

## Missing, conflicting and shifted-prior evidence

The 480 final uncertain observations have exact conditional targets obtained from their stated
independent-latent and sensor model. These probabilities are not real business event frequencies.
Three conditions are reported separately in the evidence. Shifted priors vary within the declared
range; the evaluation does not establish generalization to arbitrary unseen prior values.

| Metric (lower is better) | Foundation | Workflow |
| --- | ---: | ---: |
{chr(10).join(uncertain_rows)}

A wrong decision costs 1, missing the designated critical outcome costs 5, and a correct decision
costs 0. The cost comparison holds answer coverage at 80%. With the actual independently fitted
policy, overall final answer coverage is **{percent(policy["coverage"])}** and uncertain-request
abstention is **{percent(uncertain_policy["abstention_rate"])}**. Calibration error ≤15% at coverage
≥60% is an empirical calibration condition, not a bound on risk after distribution shift.

The actual frozen policy's final **expected error among accepted answers** is
**{percent(policy["expected_error"])} overall** and
**{percent(uncertain_policy["expected_error"])} on uncertain inputs**. These final errors are
not constrained to the calibration limit; reduced cost does not establish low absolute risk.
The reported overall 15-bin ECE is **{percent(baseline["overall"]["ece_15_bins"])}** for Foundation
and **{percent(final["overall"]["ece_15_bins"])}** for Workflow. NLL/Brier and cost improvements
must be read alongside this calibration diagnostic and the family-level accuracy results.

## Previously inspected regression checks

The official 2,000-question typed-decisions benchmark scores
**{percent(official["overall"]["accuracy"])}**.
This counts all raw predictions, including abstentions. Under the frozen deployed policy,
answer coverage on this benchmark is **{percent(official["overall"]["coverage"])}** and accuracy
among accepted answers is **{percent(official["overall"]["accepted_accuracy"])}**. Raw accuracy
and the fraction of requests receiving an automatic decision are different measurements.
Foundation scored 46.85%; Dynamic v2 scored 49.05%. Previously measured LAYA base and benchmark
specialist scored 35.95% and 76.95%. Those historical measurements used the released LAYA input
limits; 34 base-model state texts were truncated. Different training regimes prevent a universal
model ranking. This checkpoint adapted only on the recorded upstream training partition.
The official benchmark measures agreement with teacher labels, not observed business outcomes.

Full prior synthetic-policy accuracy is **{percent(legacy["by_modality"]["text"]["accuracy"])}** for
text and **{percent(legacy["by_modality"]["image"]["accuracy"])}** for images.

| Foundation corpus regression | Questions with hard labels | Foundation | Workflow |
| --- | ---: | ---: | ---: |
{chr(10).join(regression_rows)}

These older final sets are regression checks and were excluded from checkpoint selection.
TrashNet uses photographed waste, often on plain backgrounds; it is not general visual reasoning.
Per-domain changes, final uncertainty conditions and risk curves remain in `evidence/`.
Earlier failed experiments remain in the public project reports.

## Run and measured scope

This is a custom adapter/readout, not a standalone Transformers AutoModel. Install the bundled
wheel with Python 3.12 and CUDA PyTorch, then load the pinned base separately:

```sh
pip install runtime/{args.wheel.name} --extra-index-url https://download.pytorch.org/whl/cu128
python load_veyra.py --request sample_request.json
veyra serve --checkpoint . --image-root /absolute/path/to/images
```

The local service accepts POST `/v1/systemone` at `127.0.0.1:8000`. It is not a drop-in JEV SDK.
One image, 1–4 questions, 2–16 candidates per choice/score question and 2,048 total processed tokens
are supported. Inputs exceeding limits are rejected. Resident HTTP p95 is
**{acceptance["details"]["runtime"]["p95_ms"]:.2f} ms** on RTX 4060 Ti 8 GB:
40 serial distinct photos,
one question and six candidates, three warmups, including image decode and HTTP/JSON. Model
loading, concurrent load and WAN transport are excluded; other workloads need separate timing.

{architecture_note(manifest, parameters)}
A three-seed CE/proper-scoring/RLOO comparison preceded two declared backbone
learning rates and half/full checkpoints. Candidate selection used merged BF16 development
inference. Probability fitting and abstention used disjoint calibration groups. Weights,
calibration, evaluators and the final corpus were frozen before final evaluation.
{continuation_note}

## Roadmap and evidence

Priorities 1 (workflow transfer) and 2 (uncertainty and abstention) are mandatory and passed only
within the stated scope. Priority 3 — longer context, OCR, fine visual relationships and Korean —
remains future work. [Roadmap](ROADMAP.md) · [한국어 로드맵](ROADMAP.ko.md).
`release-protocol.json`, `release-readiness.json` and hash-bound `evidence/` contain the criteria
and results. Publication also requires the passing, checksum-bound report from real text/image
inference using the bundled wheel. The upload helper takes its path with `--verification-report`
and checks it before using an explicitly supplied HF target. This local verification uses the
existing pinned dependencies and cached Qwen base; it is not a clean dependency installation.

Code and trained components: Apache-2.0. Qwen: Apache-2.0. Dataset attributions and original
licenses are in `NOTICE` and `data-licenses/`. No original dataset records, photographs or feature
caches are bundled. `release-readiness.json` lists SHA-256 fingerprints for data inputs verified
locally; obtain those inputs through the repository's data builders and attributed sources.
Project: https://github.com/ken-jo/veyra
"""
    (output / "README.md").write_text(card, encoding="utf-8", newline="\n")
    checksums = {
        p.relative_to(output).as_posix(): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write(output / "checksums.json", checksums)
    write(
        output.parent / (output.name + "-preparation.json"),
        {
            "prepared": True,
            "published": False,
            "files": len(checksums) + 1,
            "folder": str(output.resolve()),
            "weights_sha256": acceptance["weights_sha256"],
            "manifest_sha256": acceptance["manifest_sha256"],
            "checksums_sha256": digest(output / "checksums.json"),
            "original_acceptance_sha256": digest(report_path),
        },
    )
    print(json.dumps({"prepared": str(output.resolve()), "published": False}), flush=True)


UPLOAD_HELPER = '''"""Publish a verified workflow package to an explicitly chosen HF repository."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--public", action="store_true")
parser.add_argument("--verification-report", type=Path, required=True)
args = parser.parse_args()
folder = Path(__file__).resolve().parent

def checked(name, expected):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Package evidence mismatch: " + name)
    return path

checksums = json.loads((folder / "checksums.json").read_text(encoding="utf-8"))
for name, expected in checksums.items():
    checked(name, expected)
readiness = json.loads((folder / "release-readiness.json").read_text(encoding="utf-8"))
required = {"workflow_generalization", "uncertainty_probability", "uncertainty_abstention",
            "retention_development", "legacy_text", "legacy_image", "official_workflow",
            "runtime", "single_network", "provenance"}
if readiness.get("release_allowed") is not True or readiness.get("required_priorities") != [1, 2]:
    raise SystemExit("Release blocked: both mandatory priorities must pass")
if any(readiness.get("checks", {}).get(key) is not True for key in required):
    raise SystemExit("Release blocked: missing or failed evidence")
manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
extensions = manifest.get("training", {}).get("release_extensions", {})
if readiness.get("release_extensions", {}) != extensions:
    raise SystemExit("Release blocked: study extension mismatch")
extension_checks = {
    "workflow_workspace_v12": (
        "workspace_study_provenance", "previous_calibration_policy_regression"
    ),
    "workflow_curriculum_v12": (
        "curriculum_study_provenance", "previous_calibration_policy_regression"
    ),
    "workflow_depth_v12": (
        "depth_study_provenance", "previous_calibration_policy_regression"
    ),
    "workflow_depth_policy_v12": (
        "depth_study_provenance", "conservative_policy_study_provenance",
        "independent_policy_validation", "previous_calibration_policy_regression"
    ),
    "workflow_cohort_policy_v12": (
        "depth_study_provenance", "cohort_policy_study_provenance",
        "independent_policy_validation", "previous_calibration_policy_regression"
    ),
}
for extension in extensions:
    if extension not in extension_checks or any(
        readiness.get("checks", {}).get(key) is not True
        for key in extension_checks[extension]
    ):
        raise SystemExit("Release blocked: continuation study requirements failed")
if any(value is not True for value in readiness.get("checks", {}).values()):
    raise SystemExit("Release blocked: an additional mandatory requirement failed")
for name, key in (("head.safetensors", "weights_sha256"), ("manifest.json", "manifest_sha256"),
                  ("release-protocol.json", "release_protocol_sha256")):
    checked(name, readiness[key])
if not readiness.get("evidence_files"):
    raise SystemExit("Release blocked: no evidence files")
for name, expected in readiness["evidence_files"].items():
    checked(name, expected)
original = json.loads(checked(
    readiness["original_acceptance_report"], readiness["original_acceptance_sha256"]
).read_text(encoding="utf-8"))
mapping = readiness.get("evidence_mapping", {})
external = readiness.get("external_input_files", {})
original_files = original.get("evidence_files", {})
if (
    readiness.get("original_dataset_inputs_bundled") is not False
    or set(mapping) & set(external)
    or set(mapping) | set(external) != set(original_files)
    or set(mapping.values()) != set(readiness["evidence_files"])
):
    raise SystemExit("Release blocked: incomplete packaged/external evidence inventory")
for source, target in mapping.items():
    if readiness["evidence_files"][target] != original_files[source]:
        raise SystemExit("Release blocked: relocated evidence changed")
for source, row in external.items():
    path = PurePosixPath(row.get("repository_path", ""))
    if (
        row.get("sha256") != original_files[source]
        or not path.parts or path.parts[0] not in {"data", "datasets"}
        or ".." in path.parts or path.is_absolute()
    ):
        raise SystemExit("Release blocked: invalid external data fingerprint")
verification = json.loads(args.verification_report.read_text(encoding="utf-8"))
wheels = list((folder / "runtime").glob("*.whl"))
if len(wheels) != 1:
    raise SystemExit("Release blocked: require exactly one bundled wheel")
if not (
    verification.get("passed") is True
    and verification.get("python_isolated") is True
    and verification.get("weights_sha256") == readiness["weights_sha256"]
    and verification.get("manifest_sha256") == readiness["manifest_sha256"]
    and verification.get("checksums_sha256")
    == hashlib.sha256((folder / "checksums.json").read_bytes()).hexdigest()
    and verification.get("wheel_sha256") == hashlib.sha256(wheels[0].read_bytes()).hexdigest()
    and set(verification.get("results", {})) == {"text", "photograph_three_questions"}
    and all(
        row.get("backbone_forwards") == 1
        and row.get("response", {}).get("usage", {}).get("output_tokens") == 0
        for row in verification["results"].values()
    )
):
    raise SystemExit("Release blocked: missing or mismatched bundled-wheel inference verification")
from huggingface_hub import HfApi, get_token
if not get_token():
    raise SystemExit("Run hf auth login first; never place a token in source files")
api = HfApi()
api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=folder,
                  allow_patterns=[*checksums, "checksums.json"],
                  ignore_patterns=[".cache/**", "__pycache__/**", ".git/**"],
                  commit_message="Publish evaluated Veyra workflow checkpoint")
print("https://huggingface.co/" + args.repo_id)
'''


if __name__ == "__main__":
    main()
