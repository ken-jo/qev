# QEV and LAYA: final release comparison

Measured on 2026-10-01. All three frozen models answered the **same 2,800 English
questions** on one RTX 4060 Ti 8 GB. No weights, temperatures or prompts were tuned
after inspecting these results. Raw prediction hashes, source revisions, input audits
and aggregate results are in [the evidence directory](https://github.com/ken-jo/qev/blob/main/reports/release-comparison/README.md).

![Accuracy on identical inputs](https://raw.githubusercontent.com/ken-jo/qev/main/reports/release-comparison/accuracy.png)

## Accuracy: higher is better

| Test | Questions | LAYA English | LAYA Typed Decisions | QEV 0.1.1 |
| --- | ---: | ---: | ---: | ---: |
| Official typed-decisions | 2,000 | 36.05% | 76.95% | 77.00% |
| AG News, 4 candidates | 400 | 95.00% | 95.25% | 82.50% |
| DAIR Emotion, 6 candidates | 400 | 58.75% | 60.00% | 50.25% |

On typed-decisions, QEV answers one more question correctly than the specialist:
1,540 versus 1,539. The paired group-bootstrap 95% interval for the difference is
**-1.85 to +1.95 percentage points**. This does not demonstrate an accuracy advantage.
Relative to the specialist, QEV is 12.75 points lower on news and 9.75 points lower on
emotion; the respective descriptive intervals are -16.75 to -9.00 and -14.75 to -4.75
points. These intervals use 10,000 paired group resamples and are not adjusted for
multiple comparisons.

## Which results are zero-shot?

| Test | QEV adaptation exposure | LAYA exposure | Interpretation |
| --- | --- | --- | --- |
| typed-decisions | Trained on the recorded training partition | Specialist fine-tuned on this benchmark; base not specialized | Adapted regression comparison |
| AG News | No source or matching test text in released adaptation snapshots | In LAYA's training mix according to its benchmark script | QEV task transfer; unequal training exposure |
| DAIR Emotion | No source or matching test text in released adaptation snapshots | Reported held out by LAYA's benchmark script | Task-held-out transfer for both |

Here **zero-shot means no QEV task-specific adaptation, examples in the prompt, or
test-time fitting**. The audit covered 15 released corpus snapshots and 251,457 rows,
including repeated stages. It found no source IDs for AG News or Emotion and no
normalized test-text substring matches. This does not prove absence from upstream
Qwen or ModernBERT pretraining, undistributed intermediate material, or semantic near
duplicates. The two 400-example suites are a limited English transfer study.

LAYA's [application benchmark source](https://github.com/NandhaKishorM/laya/blob/main/research/scripts/bench_apps.py)
uses the first 400 test examples for these tasks. We retain that selection and its
instructions. Emotion's null candidate descriptions are expressed as their label names
for both SDKs. Both models receive the same serialized text and question dictionaries.
The resulting scores are our measurements, not copied from LAYA's published tables.

## Probability quality on typed-decisions: lower is better

| Metric | LAYA English | LAYA Typed Decisions | QEV 0.1.1 |
| --- | ---: | ---: | ---: |
| Brier / squared error against soft targets | 0.31545 | 0.06149 | 0.07460 |
| NLL / cross-entropy against soft targets | 1.34337 | 0.88445 | 0.90894 |
| ECE, 15 bins | 17.54% | 21.67% | 25.19% |
| Ordinal expectation MAE | 0.69457 | 0.24251 | 0.30366 |

The specialist has lower probability and ordinal errors than QEV on this test.
Accuracy and calibration are different: a model can pick the right candidate while
assigning poorly calibrated probabilities. QEV's fitted abstention accepts 30.80% of
these questions, at 94.32% accepted accuracy. **The 77.00% headline includes all 2,000
argmax decisions, including abstentions.** The accepted subset is not substituted for
the full-test denominator. LAYA is scored without an added abstention threshold.

For completeness, QEV's news/emotion ECE is 6.68%/9.20%; the base is 2.95%/32.07% and
the specialist is 15.50%/20.08%. Lower ECE on emotion does not erase QEV's accuracy gap.
Full NLL and Brier values for both transfer tasks are included in the JSON reports.

### Decision types and workflows

| Type | Questions | LAYA Typed Decisions | QEV 0.1.1 |
| --- | ---: | ---: | ---: |
| choice | 600 | 73.50% | 73.50% |
| score, probability argmax | 800 | 72.88% | 74.75% |
| noul | 600 | 85.83% | 83.50% |

| Workflow | Questions | LAYA Typed Decisions | QEV 0.1.1 |
| --- | ---: | ---: | ---: |
| Agent trace observability | 500 | 73.40% | 70.80% |
| Customer service | 500 | 76.80% | 81.40% |
| Invoice processing | 500 | 80.60% | 83.80% |
| Security incidents | 500 | 77.00% | 72.00% |

## Response time on the same GPU: lower is better

Resident serial SDK calls, one question per call, four CPU threads and three warmups
per suite. No concurrent model server was running. These measurements include each
SDK's request processing but exclude model loading, HTTP, WAN and queueing.

| Test | LAYA base p50 / p95 | LAYA specialist p50 / p95 | QEV p50 / p95 |
| --- | ---: | ---: | ---: |
| typed-decisions | 24.22 / 29.00 ms | 23.71 / 30.06 ms | 77.70 / 97.70 ms |
| AG News | 21.50 / 24.86 ms | 27.72 / 34.59 ms | 52.34 / 61.24 ms |
| Emotion | 21.61 / 25.17 ms | 22.90 / 30.60 ms | 50.72 / 56.11 ms |

QEV is about 3.3 times slower at typed-decisions p50 in this run. It still falls within
the project's 300 ms resident response-time target on these workloads; this is not a
concurrent-service SLA. LAYA uses its released CUDA path; QEV uses BF16 backbone weights
and FP32 readouts. Optional kernel acceleration was not added to either installation.

## Vision and model size

The tested LAYA checkpoints accept text. QEV also accepts an image with textual criteria.
Its separately measured CIFAR-10 guard accuracy is 95.83% on 600 questions, and resident
local-photo HTTP p95 is 114.94 ms on the same GPU class. Those are **separate image
evaluations**, not evidence of superiority over a text-only model. The negative visual
results remain relevant: 0/72 exploratory 2048 wins and 5/28 balanced image board-cell
decisions. See [evaluation limits](https://github.com/ken-jo/qev/blob/main/docs/EVALUATION.md).

LAYA's English models are reported as 421M parameters. QEV's merged inference model
has 2,213,418,880 parameters, about 5.3 times as many. Its 32 MB adaptation file is not
the full model. See [exact counts and dtypes](MODEL_SIZE.md).

## Frozen protocol and metric definitions

- Dataset: `LocalLLaMA/typed-decisions`, revision
  `d51d993547ad8355b1c25157fbc1fea0649e8ffa`, all 400 official test cases / 2,000 questions.
  Gold labels are teacher annotations rather than independently observed outcomes.
- LAYA SDK: 0.3.20. Base revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`;
  specialist revision `1a793eb568e6718f15941d08f85432581df534e3`.
- LAYA context: 1,024 tokens for both checkpoints; released option-head budgets and
  temperatures retained. The base's stock 512-token context would truncate 34 official
  questions. The actual comparison has **zero state, instruction and option truncations**.
  QEV uses its released 2,048-token limit and rejects over-budget requests.
- Accuracy uses probability argmax against explicit hard gold. Brier sums squared
  candidate-probability differences against the target distribution. NLL is target
  cross-entropy, with probabilities clipped at 1e-12. ECE uses 15 equal-width bins
  against hard correctness. Score MAE compares expected zero-based ordinal levels.
- The same frozen metric implementation processes all predictions. LAYA's published
  SDK rounds probabilities to four decimals; returned distributions are normalized
  before scoring. SDK defaults for criteria-free noul questions remain SDK-specific.
- No multilingual router, full 77-class BANKING77, 51-language sweep or live JEV API
  comparison is claimed. QEV currently supports at most 16 candidates and four questions.

**Conclusion:** QEV adds direct image inputs to a LAYA-inspired decision interface and
matches the specialist's aggregate accuracy on this adapted text benchmark within the
measured uncertainty. LAYA remains smaller and faster here, with stronger probability
quality on typed-decisions and higher accuracy on both transfer suites. These findings
close the current Qwen-based research release and guide the compact-model roadmap.
