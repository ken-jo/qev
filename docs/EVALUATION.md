# Evaluation and limits

For the final same-input QEV/LAYA comparison and the separate task-held-out evaluation,
see [LAYA comparison](LAYA_COMPARISON.md). Benchmark adaptation, unseen task families,
and zero-shot task transfer are reported separately.

The release preserves the v13 weights and calibrated manifest. Full numerical evidence:
[fresh final](../reports/workflow-v13/final-evaluation/README.md) and
[acceptance](../reports/workflow-v13/final-evaluation/release-acceptance.json).

The [model card](../MODEL_CARD.md) presents the main results. Accuracy is higher-is-better.
NLL, Brier/squared distribution error, ECE and ordinal MAE are lower-is-better. Their scales
depend on the definition and candidate count, so comparisons require matching inputs,
targets, aggregation and normalization. ECE is the gap between confidence bins and observed
accuracy; it is not the proportion of all incorrect predictions.

## Claims we can support

The new procedural workflows improved over the recorded Foundation v11 (Qwen3.5-2B), while the
declared retention, latency and recovery acceptance gates passed. The original acceptance
contains 14 checks bound to source/evidence hashes. The selected model used development
data for selection, later separate calibration/policy populations, then new final groups.
Previously consumed final groups are marked as regressions.

## Claims we cannot support

The synthetic image result is not unrestricted real-photo accuracy. A CIFAR-10 guard is
not an OCR or spatial-reasoning benchmark. Sampled eight-candidate BANKING77 is not the
standard 77-class task. Fast serial HTTP is not concurrent throughput or an internet SLA.
No matched JEV benchmark was run. LAYA is text-only in the reference experiment, so no
LAYA photo baseline exists. Its task-specialized model has different training exposure.

