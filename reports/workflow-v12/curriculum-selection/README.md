# Matched prerequisite curriculum: merged development screen failed

All four trained checkpoints and the parent were evaluated with merged BF16 inference. Every
candidate failed the required score coverage of at least 60% under at most 15% expected error.
The evaluation coordinator and release pipeline stopped at selection. Fresh calibration and
final inference were not started. These are development results, not an independent final claim.

## Complete candidate comparison

| Candidate | New workflow accuracy | Familiar workflow accuracy | Uncertainty NLL | Cost at 80% coverage | Score coverage under 15% error |
|---|---:|---:|---:|---:|---:|
| Parent | 73.11% | 77.17% | 0.96240 | 1.30693 | 20.63% |
| Curriculum, epoch 1 | 72.67% | 75.83% | 0.87239 | 0.83256 | 22.34% |
| Outcome replay, epoch 1 | 73.89% | 74.00% | 0.85827 | 0.73833 | 18.28% |
| Curriculum, epoch 2 | 76.44% | 79.50% | 0.87644 | 0.86335 | 29.38% |
| Outcome replay, epoch 2 | 75.11% | 78.67% | 0.86318 | 0.87717 | 24.22% |

The existing ranking places curriculum epoch 2 first, but `eligible` is false. It is not an
approved release checkpoint. At 60% score coverage, its expected error is 21.54%; the second
control epoch has 22.90%, both above 15%. The curriculum's choice/noul coverage is 98.76%/83.45%.
Its photo, NLI and intent development accuracies are 96.50%, 86.89% and 83.98%.

## Matched differences and uncertainty

The difference below is curriculum minus control at the same epoch. Intervals use 2,000 paired
observation-group bootstrap replicates, seed 191. New workflows have 900 questions/300 groups;
familiar workflows have 600 questions/120 groups.

| Epoch | New workflow difference, 95% interval | Familiar workflow difference, 95% interval |
|---|---:|---:|
| 1 | -1.22 pp [-3.00, +0.56] | +1.83 pp [-0.83, +4.33] |
| 2 | +1.33 pp [-0.11, +2.78] | +0.83 pp [-0.67, +2.33] |

All intervals include zero. These measurements do not establish a curriculum accuracy advantage
over matched additional outcome training. The control has lower uncertainty NLL and squared
distribution error at both epochs; cost favors control in epoch 1 and curriculum in epoch 2.
Probability/cost comparisons here are point estimates without bootstrap intervals.

There is only one matched training seed. These intervals resample observation groups, not
workflow families or training seeds, and have no multiple-comparison correction. The development
data has been repeatedly inspected. Larger gains on the prerequisite diagnostic questions do
not establish causally faithful internal reasoning or final workflow generalization.

## Score diagnosis

For curriculum epoch 2, score hard-label accuracy is 76.67% over 540 hard questions. The score
population also contains 100 conditional-distribution questions. On complete new workflows,
score accuracy is 76.00% and mean confidence 88.20%; on familiar workflows these are
77.50% and 53.75%. All 47 hard errors with confidence at least 0.8 occur in the new-workflow
subset. This motivates examining confidence ordering across populations alongside correctness;
the observational difference does not by itself identify a causal mechanism. Annotation and
training-target semantics also differ between these populations.

At 60% coverage, an ideal ordering that knows the errors of the same predictions would have
zero error. This is an annotation-aware diagnostic, not a fitted policy or achieved performance.
Within the new/familiar score subsets, their own 60% prefixes have 15.00%/14.58% error; these are
different subsets from the pooled 60% prefix and do not constitute a pooled gate pass.

The equivalent choice/score views disagree on 32 of 300 complete-evidence groups, down from
60 for the parent, while 64 groups remain wrong in both views. Agreement alone cannot replace
correctness. Remaining investigation concerns both the model's rule decisions and the ordering
of their reported confidence. No confidence method was selected by these diagnostics.

## Evidence and next work

`manifest.json` binds 13 copied aggregate artifacts: the baseline, five candidates, selection
protocol/result, matched comparison, training completion and process outcomes. The comparison
recomputed metrics and risk curves from the saved raw logits and checked model/data/source
identities. Raw logits, training rows, optimizer state and model weights are not included here.

The [unchanged study declaration](../../../configs/workflow-curriculum-study-v12.json) and
[training/diagnostic history](../skill-curriculum/README.md) preserve both matched epochs.
No candidate, final population or numerical release threshold was changed after this result.
The next preparation examines whether allowing earlier backbone layers to learn can address
the remaining errors, with a matched control and a memory feasibility check before declaring
training. No result or advantage for that proposed change exists yet.
