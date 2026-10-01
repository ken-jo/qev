# One policy constrained across three calibration populations

**Completed; release rejected because legacy text accuracy is 89.70%, below 90%.** The selected depth-study weights
are unchanged. [Declared design](../../../configs/workflow-cohort-policy-v12.json)
· [Previous failed policy validation](../policy-calibration/README.md).

## Completed final evaluation

The cohort-policy model completed its final evaluation but **cannot be released**.
[Completed results and failures](completed-results.json) preserve the evidence.

| Check | Result |
| --- | --- |
| Previously uninspected workflow, 3 families / 480 groups | Foundation 45.35% → 71.60%; +26.25 points, paired 95% interval +21.94 to +30.63 |
| Missing/conflicting/shifted-prior probabilities | NLL 1.0286 → 0.8167; Brier 0.3354 → 0.1893; cost at 80% coverage 1.7939 → 0.9419 |
| Official, previously inspected workflow regression | 77.20%; deployed coverage 38.65%, accepted accuracy 93.79% |
| Legacy synthetic text / image | **89.70% (fails the 90% gate)** / 97.05% |
| RTX 4060 Ti resident photo HTTP | p95 111.50 ms; one backbone forward and zero generated answer tokens |

The supplemental audit passed 13 of 14 checks. Two auditor defects were repaired separately:
calibration replacement was mistaken for a changed development set, and a correct numeric
probability difference of 0.0 was treated as false. Original outputs and failure reports remain
unchanged. No performance limit was relaxed. Actual final accepted-answer expected error was
18.46% overall and 33.45% on uncertain inputs; overall ECE increased from 5.88% to 7.63%.
The calibration 15% criterion is not a guaranteed error bound for new tasks.

The next [readout recovery study](../../../configs/workflow-readout-recovery-v13.json) keeps the current
backbone frozen and compares five Foundation/current head interpolations with six replay
checkpoints. Its 9,129 training questions and 5,102 development questions are separated.
Selection requires 91% legacy text and image development accuracy plus workflow/probability
retention, followed by actual merged inference. These study screens do not replace the fixed
90% release gate. Previously inspected final groups become regression evidence; fresh final
groups are mandatory for a subsequent independent claim. Both roadmap priorities remain required.

## Independent validation results

Both new populations passed with the same frozen policy. No parameter was refitted between them.
The original-group regression also passed. Subsequent final evaluation failed legacy text;
workflow HF preparation did not begin. These validation results do not waive that failure.

| Population | Type | Answer coverage | Expected error |
| --- | --- | ---: | ---: |
| Seed 233 | choice | 94.54% | 9.82% |
| Seed 233 | score | 90.56% | 9.99% |
| Seed 233 | noul | 79.09% | 9.62% |
| Seed 239 | choice | 95.02% | 10.06% |
| Seed 239 | score | 90.42% | 12.28% |
| Seed 239 | noul | 77.88% | 11.83% |

The two `validation-*-policy.json` reports include descriptive observation-group bootstrap
intervals. These are independent policy-validation results within the declared generated/public
populations, not the original final workflow comparison or arbitrary deployment-risk guarantees.

## Previously inspected original-group regression

The same policy passed the exact original policy groups without refitting:

| Type | Answer coverage | Expected error |
| --- | ---: | ---: |
| choice | 87.39% | 7.73% |
| score | 73.91% | 9.05% |
| noul | 66.67% | 4.41% |

The earlier depth policy's score error was 16.95% at 99.73% coverage. The current policy reduces
accepted-answer error by abstaining more; it uses identical model weights and does not increase
raw argmax accuracy. These original groups were previously inspected and are regression evidence.
`previous-calibration-policy.json` records their identities, metrics and unchanged-policy checks.

## Motivation and adaptation disclosure

The first depth calibration failed the previous-group score regression at 16.95% error.
A 12% fitting target on another sample passed its fitting gates, but failed independent noul
validation at 17.09% error. That failure and its subgroup diagnosis are preserved. The difference
between populations does not establish a causal explanation; authored familiar-workflow noul
errors were high in both and higher in the failed validation.

This adaptive study explicitly reuses all three previously inspected calibration populations
as fitting data: **7,932 questions / 3,252 observation groups**. The failed validation is now
fitting data, and is never represented as independent evidence for this new policy. No original
final prediction has been used to choose this method or the model weights.

## Method and fixed sequence

1. Preserve the original development-selected full-depth epoch-2 weights.
2. Pool one group partition for the existing per-type temperature fit (minimum temperature 1).
3. Fit one threshold per type on the other partition. Scan the fixed grid 0, 0.005, ..., 1.
   Every fitting population separately must achieve **error ≤12% and coverage ≥60%**. Choose
   the lowest eligible threshold, which maximizes coverage among eligible grid points.
4. Freeze the calibrated policy and inference/audit sources.
5. Evaluate two newly declared populations, seeds 233 and 239, each with **2,644 questions /
   1,084 groups**. Every type in **both** must pass the unchanged **error ≤15% / coverage ≥60%**
   limits without refitting. Both populations passed independent target and separation audits.
6. Apply the same policy to the exact original failed policy groups, with the same limits.
7. Only after these stages pass, run the original final comparison, regressions, ≤300ms resident
   photo HTTP measurement, network contract, acceptance and actual bundled-wheel HF verification.

Cohort identity is used only in offline fitting. Deployment retains three scalar thresholds,
the existing temperatures, one network pass and zero generated answer tokens. This changes
answer selection and probability calibration; it does not train new model weights or raise
raw argmax accuracy by itself.

The idea of checking groups separately is related to [group distributionally robust optimization](https://arxiv.org/abs/1911.08731).
This experiment applies empirical constraints to calibration populations and does not reproduce
that paper's neural-network training method. These cohorts are generated samples, not a complete
set of deployment domains. No future-risk or distribution-free guarantee is claimed.

## Provenance and limitations

Every non-calibration record remains identical to the original data after serialization.
Fresh validations exclude prior populations and each other by observation groups, normalized
non-image states and image byte hashes, with perceptual duplicate screening during selection.
Each new population's 2,040 procedural targets passed independent rule reconstruction; its
604 other targets retain public source labels. Familiar workflows use authored explicit rules,
not new public teacher annotations. The original-group and official workflow regressions remain
mandatory. Public-data overlap with Qwen pretraining is unknown.

A mutable-list alias in the initial data combiner attempted to find a new calibration image
inside the original image folder. It failed before writing completed records or running a model.
The [declared data repair](../../../configs/workflow-cohort-policy-data-repair-v12.json) copies
the original list before extending it and verifies partial image hardlinks. The failed source,
failure report and repaired-source hashes are preserved; sampling, targets and policy rules did
not change. The combined records and both validation target audits passed afterward.

`manifest.json` binds aggregate evidence. Original records, photographs, logits and checkpoint
files remain local. `scripts/run_cohort_policy_release.py` stops on a failing requirement.

## Supplemental audit provenance

The original auditor stopped on whole-file dataset equality despite identical train/dev/test
records and an explicitly changed calibration subset. The first supplemental audit verified
those exact preserved records but inherited a second defect: `all(protocol.values())` treated
the valid numeric difference `0.0` as failure. The second supplemental audit requires the four
boolean checks and verifies the numeric difference against the existing `<1e-5` limit.
The original source, final outputs, first report and both repair declarations are preserved.
[Current supplemental acceptance](release-acceptance-supplemental.json) confirms only
`legacy_text` fails. [First supplemental report](release-acceptance.json) retains the spurious
runtime failure for the record. Neither repair changes a performance threshold or model output.
