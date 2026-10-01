# QEV 0.1.1

Final release of the current Qwen-based research model and its Python SDK.
QEV is inspired by LAYA's typed decisions and built with Qwen3.5-2B's text and vision
backbone, language adapters and trained decision heads. The vision encoder is frozen.

## Same-input comparison

All three frozen models were evaluated on the same 2,800 English questions on one
RTX 4060 Ti 8 GB, without new training, prompt selection or calibration fitting.

| Test | LAYA English | LAYA Typed Decisions | QEV 0.1.1 |
| --- | ---: | ---: | ---: |
| Official typed-decisions, 2,000 questions | 36.05% | 76.95% | 77.00% |
| News topic, 400 examples | 95.00% | 95.25% | 82.50% |
| Emotion, 400 examples | 58.75% | 60.00% | 50.25% |

Typed-decisions is an adapted regression benchmark. The one-question lead over the
specialist does not establish an advantage. News and emotion are task-held-out relative
to the audited QEV adaptation sources; backbone pretraining overlap is unknown.
LAYA was smaller and faster, with better probability quality on typed-decisions.
Full metrics, confidence intervals, training exposure and source hashes are published.

The model introduction includes an actual photograph with all three typed answers,
downloadable inputs and recorded outputs, and coverage-versus-accuracy tables. The example
is a previously inspected verification fixture, not a new benchmark claim.

## Downloads and installation

- Model: https://huggingface.co/ken-jo/qev
- Data: https://huggingface.co/datasets/ken-jo/qev-data
- `qev-0.1.1.zip`: checkpoint, custom runtime and evaluation evidence.
- `qev-data-0.1.1.zip`: 15 historical corpus snapshots with source-specific licenses.
- `qev-0.1.1-py3-none-any.whl` and `qev-0.1.1.tar.gz`: Python SDK distributions.

The SDK contains code. `qev download` retrieves the adaptation and pinned Qwen base.
The adaptation file is 32.01 MB; the full inference model is 2.213B parameters and the
upstream weight download is approximately 4.55 GB. Python 3.12 is required.
PyPI account configuration is pending; the release wheel can be installed directly.

## Verification and limitations

173 tests passed. All 38 frozen inference modules, learned weights and calibration
remain unchanged. Packaged text and three-type photograph inference preserve fixture
answer probabilities exactly. Wheel metadata and an isolated no-dependency installation
passed; a fresh installation of all model dependencies was not performed.

Separate image/workflow results: CIFAR-10 guard 95.83%, fresh procedural workflows
69.38%, local-photo HTTP p95 114.94 ms on RTX 4060 Ti 8 GB. The latter excludes loading,
WAN and concurrency. Visual 2048 remained unsolved: 0 wins in 72 exploratory games.
Uncertainty and visual reasoning limits are documented alongside successful results.

Public documentation is English. Personal Korean notes remain archived separately.
Corpus archives retain their original bytes, record IDs and splits; historical stages
overlap. No public demo Space is created by this release.

[GitHub: ken-jo/qev](https://github.com/ken-jo/qev)
