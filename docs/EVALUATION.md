# Evaluation and limits

For the final same-input QEV/LAYA comparison and the separate task-held-out evaluation,
see [LAYA comparison](LAYA_COMPARISON.md). Benchmark adaptation, unseen task families,
and zero-shot task transfer are reported separately.

The release preserves the v13 weights and calibrated manifest. Full numerical evidence:
[fresh final](../reports/workflow-v13/final-evaluation/README.md),
[acceptance](../reports/workflow-v13/final-evaluation/release-acceptance.json), and
[2048 study](../reports/text2048/assistance-study-v1/README.md).

The [model card](../MODEL_CARD.md) presents the main results. Accuracy is higher-is-better.
NLL, Brier/squared distribution error, ECE and ordinal MAE are lower-is-better. Their scales
depend on the definition and candidate count, so comparisons require matching inputs,
targets, aggregation and normalization. ECE is the gap between confidence bins and observed
accuracy; it is not the proportion of all incorrect predictions.

## Claims we can support

The new procedural workflows improved over the recorded Veyra Foundation v11 (Qwen3.5-2B), while the
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

## 2048 result

All 72 games missed 2048. Across development and eight new evaluation seeds there were
8,423 actions and 7,002 model calls; five single-legal-move steps were forced by the engine.
Unassisted text and image variants stalled after four unchanged actions. On new seeds,
mean scores were 2.5 for either unassisted modality, 1,526 with legal moves supplied,
1,983 with next boards/empty counts, and 3,027 with immediate merge scores added.
Random legal control averaged 1,037. Highest tile across the latter assistance variants
was 512. Detailed instructions did not solve the task.

Raw image tests provided the rendered PNG and textual rules/history, without the board
matrix. Legal/next-state assistance exposes deterministic engine information and is
reported separately. Exploratory runs executed abstained suggestions; they are not
successful operation under the default acceptance policy.

The board-cell diagnostic used 28 balanced seven-choice cases. Image accuracy was 5/28,
with 26/28 abstained; text was 21/28, with 13/28 abstained. The model's image failures do
not identify their root cause: no untouched-Qwen ablation was run. This is a reason to
investigate perception and evidence binding in future work, not proof of a particular fix.
