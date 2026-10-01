# Conservative policy calibration after the depth-study failure

**Failed independent validation before final inference.** The selected full-depth
epoch-2 weights remain unchanged. Priorities 1 and 2 and every original numerical release
criterion remain mandatory. [Declared design](../../../configs/workflow-depth-policy-v12.json)
· [Previous failed regression](../depth-selection/README.md).

## Why this follow-up exists

The first fresh fit answered almost all score questions. It passed its own fitting population,
but reached 16.95% expected error on the previously inspected policy groups, exceeding 15%.
`diagnosis.json` preserves a post-hoc sensitivity analysis. Its thresholds are **not** deployed:
it motivated declaring a 12% fitting target, then generating new fitting and validation data.
This targeted additional fitting margin; it was not new training or an increase in raw accuracy.

## Completed independent validation

The frozen policy passed its fitting gates but failed the whole-population noul requirement:

| Type | Answer coverage | Expected error | Result |
| --- | ---: | ---: | --- |
| choice | 99.68% | 13.67% | Pass |
| score | 97.22% | 13.61% | Pass |
| noul | 96.36% | 17.09% | Fail |

The noul descriptive observation-group bootstrap 95% error interval is 14.30–19.91%.
`independent-policy-validation.json`, `release-status.json` and
`population-shift-diagnosis.json` preserve the result. Previous-group regression, final inference,
new runtime checks and HF preparation did not start. A [separately declared follow-up](../cohort-policy/README.md)
now treats these inspected groups as fitting data and requires two new validation populations.

## Declared sequence

1. Preserve the original development selection and exact model weights.
2. Generate separate fitting and validation populations: 2,644 questions / 1,084 observation
   groups each. Fit seed is 223; validation seed is 227. Existing training, development and final
   records stay byte-equivalent after record serialization. No model output constructs targets.
3. Fit temperatures on one group partition and thresholds on the other, retaining at least
   60% of answers with at most **12%** empirical expected error, separately for each type.
4. Freeze the calibrated policy and inference/audit sources before validation predictions.
5. Apply it unchanged to all independent validation questions. Each type must retain at least
   **60%** coverage with at most **15%** expected error. No refitting or candidate ranking occurs.
6. Pass the same limits on the exact original policy groups; disclose their prior inspection.
7. Only then freeze and run the original final comparison, retention, speed, single-network
   acceptance and isolated HF-package inference. Any failing gate stops the sequence.

Both new populations passed independent reconstruction of 2,040 procedural targets each.
The remaining 604 targets per population retain public source labels. Familiar-workflow cases
use authored explicit policies because the original public teacher training pool was exhausted;
they are not new teacher annotations or an interchangeable official benchmark. The old-group
regression and official workflow regression remain mandatory.

The lineage audit found no shared observation groups, normalized non-image states or image
bytes between fitting and validation populations. Source selection additionally screens prior
image populations for perceptual duplicates. Public-data overlap with Qwen pretraining remains
unknown; this check does not establish independence from pretraining.

## Evidence and interpretation

`manifest.json` binds every exported evidence file. It includes data protocols, independent
target audits, the diagnosis, continuation selection and lineage audit. Raw records, original
images, logits and weights are not copied here. An isolated `pyarrow==21.0.0` dependency repaired
data preparation; the failed attempt is preserved and model dependencies were not changed.
`evaluate-workflow-before-margin.py.txt` preserves the evaluator used in the preceding failure.

Descriptive observation-group bootstrap intervals accompany policy validation. Soft
targets and related questions within groups do not justify a binary IID risk guarantee.
The empirical margin does not guarantee future error under distribution shift. Rejection-based
coverage/error tradeoffs are discussed in [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500).
Formal procedures such as [Learn then Test](https://arxiv.org/abs/2110.01052) require their own
assumptions and testing construction; this experiment does not claim those guarantees.

Run coordinator: `scripts/run_depth_policy_release.py`. The original final population has not
been used to choose these weights, thresholds or the new fitting target.
