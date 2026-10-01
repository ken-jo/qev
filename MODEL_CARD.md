---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
base_model_relation: adapter
library_name: pytorch
language:
- en
- ko
tags:
- multimodal
- dynamic-classification
- qwen3_5
- calibrated-decisions
- custom-runtime
datasets:
- ken-jo/qwen3.5-classification-data
- LocalLLaMA/typed-decisions
- stanfordnlp/snli
- PolyAI/banking77
- AI-Lab-Makerere/beans
- garythung/trashnet
inference: false
---

# Qwen3.5 Classification — Qwen3.5-2B typed decisions

[Source code](https://github.com/ken-jo/qwen3.5-classification) ·
[Training data](https://huggingface.co/datasets/ken-jo/qwen3.5-classification-data)

Maintainer: [GitHub](https://github.com/ken-jo) ·
[LinkedIn](https://www.linkedin.com/in/ik-chan-jo).

Version **0.1.0**, final Qwen-based research snapshot. Previously developed as Veyra
Workflow Recovery v13. This release changes the public identity and packaging, not the
learned weights, candidate encoding, calibration or inference mathematics.

## Base and modifications

- Base: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Frozen vision encoder; 24 stored language-layer LoRA adapters, rank 8, alpha 16.
- 2,048-dimensional hidden states; 16-position option readout, four internal reasoning
  slots and a rank-64 condition-modulated binding head.
- Final recovery updates only adapters in language layers 18-23 and existing readout/
  binding parameters. All 24 stored adapter layers merge for inference in BF16.
- Stored adaptation/readout: 7,992,384 parameters; 32,009,800-byte safetensors file.
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
The trained prompt marker `VeyraResult:` and internal `veyra` package remain unchanged.

## Evaluation

| Fresh final metric | Foundation reference | This checkpoint |
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

- Final overall ECE is 12.58%, worse than the Foundation reference's 6.80%.
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
licenses. The model package does not contain original third-party training records or
photos. Eligible corpus snapshots are published separately, with CIFAR-10 excluded pending
redistribution rights. Read the repository's data and training documentation.

## Integrity and loading

- Weights SHA-256: `84b57aeeb987f73416ac0f796150957d1c5beb1ed373733053e46438f77a8fee`
- Manifest SHA-256: `d83f9910196c6e801658ca3816c3d2bd4a849ba3da6ff059a0edf7c6028e898d`

The package contains a calibrated manifest, adaptation weights and a custom runtime wheel.
Qwen backbone weights must be downloaded separately at the pinned revision. Install the
wheel from `runtime/`, run `qwen3.5-classification download`, then use `load_qwen3_5_classification.py` or the Python
API in the [repository](https://github.com/ken-jo/qwen3.5-classification). A standard Transformers
auto-model loader cannot directly load this custom adapter/head layout.

Historical stage flags in the unchanged manifest record when those stages were run.
The exact-model acceptance report and publication verification are separate evidence;
packaging does not retroactively alter a stage's provenance.
