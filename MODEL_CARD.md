---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
base_model_relation: adapter
library_name: pytorch
pipeline_tag: image-text-to-text
language:
- multilingual
tags:
- multilingual
- multimodal
- dynamic-classification
- qwen3_5
- calibrated-decisions
- classification
- structured-output
- custom-runtime
datasets:
- ken-jo/qev-data
- LocalLLaMA/typed-decisions
- stanfordnlp/snli
- PolyAI/banking77
- AI-Lab-Makerere/beans
- garythung/trashnet
inference: false
---

# QEV

[![Multilingual](https://img.shields.io/badge/languages-Multilingual-2563eb)](#language-support)

**Your evidence. Your criteria. A decision with probabilities.**

QEV is an open multimodal decision model inspired by
[LAYA](https://huggingface.co/convaiinnovations/laya), combining request-defined typed
decisions with the text and vision backbone of
[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B).
Provide text, an image, or both, plus the choices or criteria you want evaluated.
Receive probabilities and a structured answer in one batched backbone forward.

[Training data](https://huggingface.co/datasets/ken-jo/qev-data)

## Highlights

- **Small and local:** 2.213B merged parameters. Measured on an RTX 4060 Ti 8 GB with the
  model resident: p50 **77.70 ms** per text question (SDK) and p95 **114.94 ms** for a
  photo question over local HTTP. Loading and network time are excluded.
- **Text, a photo, or both:** `choice`, ordered `score` and true/false `noul` in one request,
  each answered with probabilities and an abstention flag. No generated answer text.
- **Matches a specialist on its own benchmark:** 77.00% on the 2,000-question typed-decisions
  test, against 76.95% for LAYA Typed Decisions (paired 95% interval -1.85 to +1.95
  points, so a tie rather than a win).
- **Honest limits:** LAYA's specialist has better probability quality (Brier, NLL, ECE) and
  is faster on that benchmark, and QEV is behind it on zero-shot news and emotion. QEV has
  not yet been run on a public decision leaderboard; see [Limitations](#limitations-and-negative-results).

## Language support

**Multilingual inputs** use Qwen3.5-2B's multilingual text backbone. The interface and
documentation are in English. QEV's published task and calibration evaluations are
English-focused; equivalent accuracy across languages has not been established.

## At a glance

| | QEV 0.1.1 |
| --- | --- |
| Evidence | Text, one photo, or text and a photo together |
| Decision types | `choice`, ordered `score`, and true/false `noul` |
| Languages | Multilingual inputs; reported evaluations focus on English |
| Request limits | 1-4 questions; 2-16 choice/score candidates; 2,048 processed tokens |
| Inference | One batched backbone forward; zero generated answer tokens |
| Base model | Qwen3.5-2B, with its vision encoder frozen |
| Full model / adaptation | 2.213B merged parameters / 7.992M stored adapter and head parameters |
| Precision | BF16 backbone and FP32 readouts on CUDA |
| License | Apache-2.0 for code and adaptation; source-specific dataset licenses |

Model version **0.1.1**, final Qwen-based research snapshot; Python SDK **0.2.1**. The
SDK release changes packaging, not the learned weights, candidate encoding, calibration or
inference mathematics. QEV is independently maintained; its neural backbone and vision
encoder come from Qwen3.5-2B, and no LAYA checkpoint is merged into the weights. The
released recipe is supervised adaptation and calibration, not RLCD. See
[Provenance](#provenance) for the model's earlier names and internal identifiers.

## What it returns

| Primitive | Define in the request | Receive |
| --- | --- | --- |
| `choice` | Candidate names and descriptions | A probability for each candidate and a selected name |
| `score` | Ordered descriptions, such as severity levels | Level probabilities and the expected zero-based level |
| `noul` | A yes/no proposition | The probability that the proposition is true |

All three include confidence and an abstention signal. Candidate definitions can change
between requests. The output is computed directly from decision heads, so inference does
not generate an answer sentence or a JSON string that needs parsing.

## One photo, three decisions

Recognize a material, apply your own handling policy, and evaluate a proposition in the
same request. The example below is an existing release verification fixture with its
actual recorded output.

<img src="https://huggingface.co/ken-jo/qev/resolve/main/examples/photograph/item.jpg" alt="TrashNet verification photograph of a plastic bottle" width="420" />

| Question | Definition supplied with the request | QEV's recorded answer |
| --- | --- | --- |
| `choice` | Choose among cardboard, glass, metal, paper, plastic and trash | Plastic (`c4`), probability **0.9361** |
| `score` | Band 0: glass/plastic; band 1: metal/trash; band 2: cardboard/paper | Expected level **0.4254** on the 0-2 scale; most likely band 0, probability **0.6978** |
| `noul` | Does the item belong to glass, paper or plastic? | Probability true **0.7903** |

All three answers were accepted by the released abstention policy. They used one batched
forward and generated zero answer tokens. An expected level is a weighted average over
the ordered levels. The probability on a single fixture is not an accuracy measurement.

[Exact request](https://huggingface.co/ken-jo/qev/blob/main/examples/photograph/request.json)
· [Full recorded response](https://huggingface.co/ken-jo/qev/blob/main/examples/photograph/response.json)
· [Reproduction and provenance](https://github.com/ken-jo/qev/tree/main/examples/photograph)

Photo: TrashNet, Gary Thung, MIT. Image bytes and the original verification question
definitions are preserved. This previously inspected example illustrates the interface;
aggregate performance is reported below.

## Quickstart

Use Python 3.12. The checkpoint consists of an adaptation and decision heads; its Qwen
backbone is downloaded separately on first use. Tested on an RTX 4060 Ti 8 GB; CPU
inference is supported (FP32) and slower.

```sh
python -m pip install qev==0.2.1
qev playground
```

A wheel for SDK 0.2.1 is also attached to this model repository under `runtime/`.
It is a separately built artifact; its checksum differs from the PyPI wheel.

```python
from pathlib import Path
from qev import load, DecisionRequest

model = load()  # Prepares the pinned weights on first use.
request = DecisionRequest.model_validate({
    "state": {"text": "I was charged twice. Please refund the duplicate payment."},
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {"billing": "Payments and refunds", "technical": "Software faults"},
        }
    },
})
result = model.predict(request)
print(result["answers"]["department"])
```

Recorded output for this request, shortened:

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": {"billing": 0.974044, "technical": 0.025956},
  "confidence": 0.974044,
  "abstained": false
}
```

For a photo, set `state.images` to `[{"path": "item.jpg"}]`, describe the visual decision
in the question, and call `model.predict(request, Path("images").resolve())`. The image
path is resolved under that directory. Text can supply context or a policy for the same image.
Use the [API guide](https://github.com/ken-jo/qev/blob/main/docs/API.md) for score and noul
schemas and the local HTTP server.

### Request format

| Field | Description |
| --- | --- |
| `state.text`, `state.images` | Text, and at most one image given as `{"path": ...}` under the server's image root |
| `questions` | 1-4 questions keyed by an ID you choose |
| `type` | `choice` (named options), `score` (ordered options) or `noul` (true/false) |
| `instructions` | What to decide |
| `criteria` | `choice`: option ID to description (2-16); `score`: ordered list (2-16); `noul`: optional `true`/`false` descriptions |

**Unreleased; source checkout only:** `POST /v1/systemone` (from `qev serve`) also accepts
a Jev/SystemOne-style body: a plain-string or structured JSON `state`, `instructions`
omitted (the question ID is used) and `"model": "qev"`. The published SDK **0.2.1** and
the attached 0.2.1 wheel require the native request format shown above.

State objects containing `text` or `images` use strict native validation; additional
fields are rejected instead of silently discarding an image. Other JSON states are
serialized to text. This covers the request shape only; it has not been checked against
the Jev service, images remain local paths, and the response is QEV's own.

### Files

| File | Purpose |
| --- | --- |
| `head.safetensors`, `manifest.json` | Adaptation, readout heads, and the calibrated abstention policy (hashes below) |
| `runtime/qev-0.2.1-py3-none-any.whl` | Runtime, SDK, server and English playground |
| `examples/` | Photograph request, recorded response and provenance |

The SDK includes the English playground and six sample photographs. `qev playground`
opens a local server at http://127.0.0.1:7860. First use downloads about 4.6 GB of model
files; subsequent launches use the persistent cache. Use `--offline` for a prepared cache.

## Base and modifications

- Base: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Frozen vision encoder; 24 stored language-layer LoRA adapters, rank 8, alpha 16.
- 2,048-dimensional hidden states; 16-position option readout, four internal reasoning
  slots and a rank-64 condition-modulated binding head.
- Final recovery updates only adapters in language layers 18-23 and existing readout/
  binding parameters. All 24 stored adapter layers merge for inference in BF16.
- Stored adaptation/readout: 7,992,384 parameters; 32,009,800-byte safetensors file.
- Runtime backbone: 2,213,241,664 parameters (331,416,576 vision; 1,881,825,088 language).
- Storage format: safetensors. Adaptation/readout tensors are FP32. GPU inference uses
  BF16 backbone weights and FP32 readouts; CPU inference uses FP32.
- LoRA contributes 7,815,168 stored parameters, merged into existing weights at inference.
  The remaining 177,216 readout/binding parameters stay separate; merged inference has
  2,213,418,880 parameters. The upstream 4.55 GB download also includes unused MTP tensors.
- One batched backbone forward per request; zero autoregressively generated answer tokens.
  Questions are encoded as separate batch entries, so adding questions still costs compute.

Candidate descriptions define meanings at request time. The 16 output positions are
temporary option positions, not 16 fixed semantic classes. Choice descriptions are
canonically sorted; ordered score levels retain their order. The whole model still uses
the Qwen multimodal architecture. There is no claim of a newly invented base network.

## Inputs and outputs

Text and/or one image; 1-4 questions; choice/score with 2-16 alternatives, or a binary
noul proposition. Processed token budget: 2,048. Image preprocessing uses 65,536-262,144
pixels and 32-pixel alignment. This budget does not establish a minimum reliable resolution
for small text, tiny objects or spatial tasks.

Returns probabilities, selected label/expected score/probability true, and abstention.
Probabilities are estimates; a direction probability in a game is not a game-win probability.

## QEV and LAYA: matched text evaluation

All three frozen models answered the same English inputs on an RTX 4060 Ti 8 GB.
Weights, prompts and temperatures were not tuned on this evaluation.

| Test | Questions | LAYA English | LAYA Typed Decisions | QEV 0.1.1 |
| --- | ---: | ---: | ---: | ---: |
| Official typed-decisions | 2,000 | 36.05% | 76.95% | 77.00% |
| AG News, 4 candidates | 400 | 95.00% | 95.25% | 82.50% |
| DAIR Emotion, 6 candidates | 400 | 58.75% | 60.00% | 50.25% |

![Accuracy on identical inputs](https://huggingface.co/ken-jo/qev/resolve/main/release-comparison/accuracy.png)

Typed-decisions is an adapted, previously inspected benchmark for QEV and LAYA's
specialist. QEV's one-question advantage is not evidence of superiority: the paired 95%
interval is -1.85 to +1.95 percentage points. News and emotion have no QEV task-specific
adaptation in the audited release sources, making them task-held-out zero-shot tests
for QEV. Upstream pretraining overlap is unknown. LAYA reports news in its training
mix and emotion held out, so the training exposure is not identical.

| typed-decisions metric | LAYA Typed Decisions | QEV 0.1.1 | Better direction |
| --- | ---: | ---: | --- |
| Brier against soft targets | 0.06149 | 0.07460 | Lower |
| NLL / soft-target cross-entropy | 0.88445 | 0.90894 | Lower |
| ECE, 15 bins | 21.67% | 25.19% | Lower |
| Ordinal expectation MAE | 0.24251 | 0.30366 | Lower |
| Resident SDK p50 | 23.71 ms | 77.70 ms | Lower |
| Resident SDK p95 | 30.06 ms | 97.70 ms | Lower |

The specialist has better probability quality on this benchmark and is faster here.
Both LAYA English checkpoints are reported as 421M parameters; QEV uses 2.213B after
merging adapters. Timings are serial, one question per call after warmup; loading,
network transport and queueing are excluded. All state, instruction and option
truncation counts are zero. These measurements do not establish parity with LAYA's
multilingual router or a live JEV service.

[Full comparison, source revisions and zero-shot definitions](https://github.com/ken-jo/qev/blob/main/docs/LAYA_COMPARISON.md)
· [Machine-readable evidence](https://github.com/ken-jo/qev/tree/main/reports/release-comparison)

## Accuracy and the decisions accepted

QEV reports an abstention flag from its released fitted policy. An application can route
flagged requests for review. The table shows both the whole evaluation and the portion
accepted by that policy, with no threshold retuning for these measurements.

| Evaluation | Accuracy on all questions | Questions accepted | Accuracy among accepted questions |
| --- | ---: | ---: | ---: |
| Typed-decisions, 2,000 questions | 77.00% | 30.80% | 94.32% |
| AG News, 400 questions | 82.50% | 98.00% | 83.16% |
| DAIR Emotion, 400 questions | 50.25% | 63.75% | 60.78% |

Coverage and reliability change substantially across tasks. The 94.32% figure applies
only to the accepted typed-decisions subset. It is not the accuracy of all requests or
a guarantee for a new workflow. The `abstained` field is a policy decision based on model
confidence; QEV does not return a separately trained `unknown_probability` class.

## Separate image and workflow evaluation

| Fresh final metric | Veyra Foundation v11 (Qwen3.5-2B) | QEV 0.1.1 (Workflow Recovery v13) |
| --- | ---: | ---: |
| Authored workflow accuracy, 1,440 questions / 480 groups | 44.31% | 69.38% |
| CIFAR-10 guard, 600 questions | 95.17% | 95.83% |
| SNLI, 450 questions | 86.00% | 87.78% |
| BANKING77 sampled 8-candidate, 462 questions | 84.20% | 85.50% |
| Uncertainty NLL, 480 groups (lower is better) | 1.003285 | 0.899093 |
| Squared conditional-distribution error (lower is better) | 0.323918 | 0.245122 |
| Expected 0/1/5 cost at 80% coverage (lower is better) | 1.786584 | 1.011177 |

Workflow improvement has a paired group-bootstrap 95% interval of +20.07 to +30.14
percentage points. The three new families are equipment reservation, supplier onboarding
and travel reimbursement. They are procedural tasks, not broad production business cases.

Previously inspected regressions: official typed-decisions 77.00% over 2,000 questions;
legacy synthetic text 91.16% and image 97.77%. The official benchmark's actual answer
coverage is 30.80%, with 94.32% accuracy among accepted answers. These are not independent
final sets or an overall claim of 90%+ real-world accuracy. The model trained on the recorded
typed-decisions training partition; teacher-agreement annotations are not observed outcomes.

Resident local HTTP photo p95: **114.94 ms** on RTX 4060 Ti 8 GB, 40 distinct serial
photographs after three warmups, one image/question and six candidates. Feature caching
is disabled. Loading, WAN and concurrent traffic are excluded. No matched JEV speed
or accuracy comparison was run.

## Limitations and negative results

- Final overall ECE is 12.58%, worse than Veyra Foundation v11's 6.80%.
- Final overall accepted expected error is 20.49% at 88.99% coverage. For uncertain
  requests it is 45.57% at 52.50% coverage. Fitted abstention is not a shifted-domain guarantee.
- 72 exploratory 2048 games produced no 2048 wins. Engine-assisted variants supplied legal
  directions or deterministic next boards; their results must not be credited to pure vision.
- A balanced seven-choice board-cell diagnostic scored image 5/28 (17.86%) and text
  21/28 (75%). This measures narrow digit/location binding, not general photograph accuracy.
- Photographic transfer covers limited datasets. OCR, localization, object relations,
  temporal reasoning, safety-critical use and broad Korean task quality are not established.
- Public benchmark overlap in Qwen pretraining cannot be ruled out. Near-duplicate grouping
  only detects the documented image similarities; it cannot prove distinct physical objects.

## Training and licensing

Supervised decision/distribution training and staged LoRA adaptation were used. Five
probability-learning variants, including RLOO, were compared at the earlier head stage;
cross-entropy was selected there. This checkpoint is not described as a successful new
reinforcement-learning algorithm. Later recovery uses soft cross-entropy with parent KL
replay. The selected final recovery consumed 4,533 additional questions and 567 optimizer
steps (half of the declared one-pass schedule), selected on development data before new final
evaluation. Calibration and abstention use separate recorded groups.

Code/adaptation weights are Apache-2.0; dataset material retains its separate per-source
licenses. The model package includes one MIT-licensed verification photograph with its
source notice and request. Training corpus snapshots are published separately, with
CIFAR-10 excluded pending redistribution rights. Read the repository's data and training
documentation.

## Integrity and loading

- Weights SHA-256: `84b57aeeb987f73416ac0f796150957d1c5beb1ed373733053e46438f77a8fee`
- Manifest SHA-256: `d83f9910196c6e801658ca3816c3d2bd4a849ba3da6ff059a0edf7c6028e898d`

The package contains a calibrated manifest, adaptation weights and a custom runtime wheel.
Qwen backbone weights must be downloaded separately at the pinned revision. Install the
wheel from `runtime/`, then run `qev playground`, `qev predict`, or use the Python
`load()` API in the [repository](https://github.com/ken-jo/qev). A standard Transformers
auto-model loader cannot directly load this custom adapter/head layout.

Historical stage flags in the unchanged manifest record when those stages were run.
The exact-model acceptance report and publication verification are separate evidence;
packaging does not retroactively alter a stage's provenance.

## Provenance

QEV was previously developed as Veyra, and this checkpoint as Veyra Workflow Recovery v13.
The public rename did not change the weights. The trained prompt marker `VeyraResult:` and
the internal `veyra` package keep their original names because the learned model depends
on them; model ID `Qwen/Qwen3.5-2B` in the request contract identifies the actual base.

## Citation

```bibtex
@misc{qev_2026,
  title        = {{QEV}: A Calibrated Multimodal Decision Model on Qwen3.5-2B},
  author       = {{ken-jo}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/ken-jo/qev}}
}
```

---

[GitHub: ken-jo/qev](https://github.com/ken-jo/qev)
