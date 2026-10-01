# Recovery admission and independent policy validation

The [completed recovery study](../backbone-memory-repair/README.md) selected the 50% checkpoint
using the originally declared development ranking. `admission-audit.json` passed: all three
5,102-question development evaluations were reconstructed from saved merged outputs, and the
training, frozen parameters, sources and fresh-data provenance were checked against 114 evidence
fingerprints. `selection.json` binds the exact weights, development selection and data sources.

**Calibration, both independent validations and the original-group regression passed.
The fresh final evaluation has started. No release is approved.**

The [policy configuration](../../../configs/workflow-recovery-policy-v13.json) retains the
predeclared fitting method. The explicitly inspected B/C/D populations are fitting data. Each
question type receives one temperature and one threshold, with separate groups for fitting
temperature and threshold. Expected error must be at most 12% with at least 60% answer coverage
in each fitting cohort. This tighter fitting target does not replace any release threshold.

All nine type/cohort combinations passed. The two extrema in each row may come from different
cohorts; they summarize the worst fitting error and lowest fitting answer coverage separately.

| Type | Temperature | Threshold | Largest cohort expected error | Smallest cohort coverage |
| --- | ---: | ---: | ---: | ---: |
| choice | 2.37137 | 0.505 | 11.91% | 93.21% |
| score | 4.46684 | 0.480 | 11.93% | 92.56% |
| noul | 7.71792 | 0.705 | 11.73% | 73.93% |

These are fitting results on explicitly inspected populations. `policy-freeze.json` binds the
weights, calibrated manifest, fitting report, evaluation sources and both new validation corpora
before independent predictions. Weights remain unchanged; the calibrated manifest SHA-256 is
`d83f9910196c6e801658ca3816c3d2bd4a849ba3da6ff059a0edf7c6028e898d`.

The frozen policy is applied unchanged to new validation populations G and H. Each question
type must meet at most 15% expected error and
at least 60% coverage in **both** populations, followed by the same limits on the original
policy groups. No refitting on these validation results is permitted in this study.

## Independent validation results

Populations G and H each completed all 2,644 questions and passed with the unchanged frozen
policy. The original-group regression also passed, using exactly the earlier policy groups.

| Population | Type | Expected error among accepted answers | Answer coverage | Result |
| --- | --- | ---: | ---: | --- |
| G | choice | 10.11% | 94.30% | Pass |
| G | score | 10.87% | 90.00% | Pass |
| G | noul | 8.50% | 73.79% | Pass |
| H | choice | 9.59% | 95.17% | Pass |
| H | score | 12.93% | 91.67% | Pass |
| H | noul | 11.57% | 75.91% | Pass |
| Previous groups (inspected regression) | choice | 7.73% | 87.69% | Pass |
| Previous groups (inspected regression) | score | 8.78% | 69.84% | Pass |
| Previous groups (inspected regression) | noul | 2.63% | 61.26% | Pass |

The original-group check contains 666 choice, 368 score and 333 noul questions. Its noul
coverage is only 1.26 percentage points above the fixed 60% minimum. These previously inspected
groups are regression evidence, not another independent population.

These checks use the original 15% error / 60% coverage limits. The JSON also provides
observation-group bootstrap intervals as descriptive evidence; they were not used to fit
or select the policy and do not establish a distribution-free deployment-risk guarantee.
For example, H's score expected-error interval is **10.53% to 15.42%**. Its upper endpoint
exceeds 15%; the preregistered policy gate uses the point estimate, which passes. No criterion
was changed in response to that interval. The separate final NLL gate requires its paired
95% delta interval to be strictly below zero.

`pre-final-policy-checks.json` passed all four provenance and policy requirements before the
[new final evaluation](../final-evaluation/README.md) began. All original performance, latency,
network and package requirements remain mandatory. The manifest records byte-identical source
report hashes; running progress files, original observations, raw logits and weights are excluded.
