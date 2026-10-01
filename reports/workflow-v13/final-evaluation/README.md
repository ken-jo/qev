# Fresh recovery final evaluation

**All 14 exact-model acceptance checks and exact HF-package inference verification passed.
The HF package is prepared; no HF model has been published.**

The [policy prerequisites](../policy-validation/README.md) passed before final prediction.
Runtime `0.3.0a4` adds the four mandatory recovery provenance/policy checks and preserves the
original numeric gates. The wheel was compared byte-for-byte with all 38 Python source modules.
`runtime-transition.json` records the two source-file changes; their original bytes remain in
`../runtime-source-snapshots/`. No model weight or training environment was changed at this step.

`final-freeze.json` fixes the selected weights, calibrated manifest, data and evaluator sources
before Foundation or recovery predictions. The selected weight SHA-256 is
`84b57aeeb987f73416ac0f796150957d1c5beb1ed373733053e46438f77a8fee` and calibrated manifest is
`d83f9910196c6e801658ca3816c3d2bd4a849ba3da6ff059a0edf7c6028e898d`.

The new final contains 3,432 questions: 480 observation groups across equipment reservation,
travel reimbursement and supplier onboarding provide 1,440 complete-policy questions and 480
uncertain questions. There are 160 groups per missing/conflicting/shifted-prior condition.
The remaining 1,512 questions are fresh CIFAR-10, SNLI and BANKING77 observations, subject to
the [documented sampling limits](../fresh-data/README.md). The consumed v12 final is not reused
as independent evidence. Authored workflow families do not represent unrestricted business use.

## Completed results

| Fresh final metric | Foundation | Recovery | Paired 95% interval for the change |
| --- | ---: | ---: | --- |
| New workflow accuracy, 1,440 questions / 480 groups | 44.31% | 69.38% | +20.07 to +30.14 percentage points |
| Uncertainty NLL, 480 groups | 1.00329 | 0.89909 | −0.15039 to −0.05638 |
| Squared conditional-distribution error | 0.32392 | 0.24512 | −0.10995 to −0.04784 |
| Expected 0/1/5 cost at 80% answer coverage | 1.78658 | 1.01118 | −0.96670 to −0.60294 |
| Fresh photograph guard, 600 questions | 95.17% | 95.83% | −0.33 to +1.83 points |
| Fresh SNLI, 450 questions | 86.00% | 87.78% | −0.22 to +3.78 points |
| Fresh eight-candidate BANKING77, 462 questions | 84.20% | 85.50% | −0.43 to +3.03 points |

Intervals resample observation groups with 2,000 bootstrap replicates. They do not measure
training-seed variation, are not adjusted for multiple comparisons, and do not turn the familiar
domain point increases into statistically established gains.

| New workflow family, 160 groups each | Foundation | Recovery |
| --- | ---: | ---: |
| Equipment reservation | 46.67% | 68.54% |
| Supplier onboarding | 44.79% | 65.21% |
| Travel reimbursement | 41.46% | 74.38% |

Official workflow regression is **77.00%** over 2,000 questions, exceeding the fixed 49.05%
requirement. Under the actual abstention policy it answers only **30.80%**, with 94.32% accuracy
among accepted answers. This does not establish JEV/LAYA parity or broad operational coverage.

Legacy regression is **1,867/2,048 = 91.16% text** and **1,494/1,528 = 97.77% image**, both above
the fixed 90% point-estimate requirements. Text's descriptive group-bootstrap interval is
89.65% to 92.48%, which includes 90%; these previously inspected groups are regression evidence.
Compared with the failed v12 candidate, text gains 30 correct answers and image gains 11.
The historical Foundation's text result was 92.48%, so recovery still gives up 1.32 points there.

## Runtime and acceptance

RTX 4060 Ti photo HTTP p95 is **114.94 ms** over 40 distinct measured photographs, after three
warmups, using one photograph, one question and six candidates per request. Weights are resident,
feature caching is disabled, and requests are serial. Loading, WAN transport and concurrent
load are excluded. Both text and a photograph with three question types used one backbone
forward and zero generated answer tokens.

`release-acceptance.json` passes all 14 original and recovery-extension checks, binding the exact
weights, calibrated manifest, protocol and evidence. Additional Foundation regression reports
88.04% over its labeled questions; waste/leaf photography is 89.11% / 82.29%, SNLI 88.11% and
eight-candidate intent 87.34%. Those are previously inspected regressions, not fresh final claims.

## Remaining limitations

On the full final population, actual answer coverage is 88.99% and accepted-answer expected
error is 20.49%. On uncertain requests, the policy answers 52.50% and expected error is **45.57%**.
That is higher than the always-answer uncertainty error of 44.54%, despite lower expected cost.
Overall 15-bin ECE worsens from **6.80% to 12.58%**. The 15% fitting/validation requirement is
not a risk bound for this shifted final population.

Aggregate uncertainty NLL improves conclusively under the stated bootstrap; shifted-prior NLL
alone has a delta interval of −0.22660 to +0.01562, which includes zero. On the recovery's same
252 accepted uncertain requests, the conditional oracle error floor is 21.89% and additional
model decision error is 23.68 points. The [completed-final diagnostic](../uncertainty-diagnosis/recovery-final.md)
records the exact assumptions, same-request Foundation comparison and condition breakdown.
No final observation was used to alter the current model, policy or release criteria.

The coordinator completed the paired comparison, official/legacy/Foundation regressions,
actual HTTP latency, one-forward network contract and acceptance. It stopped before packaging
so result documentation could be updated. Packaging then completed with 190 files and 189
checksum entries. `hf-package-verification.json` confirms real text/image inference using the
bundled `0.3.0a4` wheel in isolated Python. Both requests used one backbone forward and zero
generated answer tokens; the photograph covered choice, score and noul. Installed dependencies
and the cached base were reused; this is not a clean-machine dependency installation.

Only completed immutable aggregates are added here. The manifest verifies their bytes; original
data, photographs, raw predictions and weights are not included in this report directory.
