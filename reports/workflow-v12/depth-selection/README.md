# Matched depth study: merged development selection passed

All five declared candidates were evaluated in merged BF16 form. Both full-depth checkpoints
passed the original development requirements and the per-type abstention screen. The fixed
ranking selected **full-depth epoch 2**. The parent and both zero-rate earlier-layer controls
failed the score coverage screen. This is development eligibility; the mandatory release gates
remain incomplete. The subsequent unchanged-policy regression failed for score questions,
and the release pipeline stopped before final evaluation.

## Complete candidate comparison

| Candidate | New workflow accuracy | Familiar workflow accuracy | Uncertainty NLL | Uncertainty Brier | Cost at 80% coverage | Score coverage under 15% error | Development eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Parent | 76.44% | 79.50% | 0.87644 | 0.21538 | 0.86335 | 29.38% | No |
| Full depth, epoch 1 | 85.44% | 77.33% | 0.83260 | 0.17214 | 0.60342 | 64.06% | Yes |
| Control, epoch 1 | 75.89% | 76.67% | 0.85380 | 0.20191 | 0.78038 | 24.69% | No |
| Full depth, epoch 2 | 89.44% | 80.67% | 0.77165 | 0.13586 | 0.59286 | 91.09% | Yes, selected |
| Control, epoch 2 | 76.89% | 78.67% | 0.87399 | 0.20824 | 0.84556 | 26.09% | No |

The selected candidate's photo, sentence-relation and intent development accuracy is
96.83%, 88.22% and 84.42%. Its choice/score/noul coverage under 15% expected error is
100.00%/91.09%/100.00%; score expected error at 60% coverage is 10.88%. These diagnostic
thresholds are evaluated on development groups; they are not the independently fitted
deployment abstention policy.

The earlier [unmerged diagnostics](../depth-study/README.md) remain available, including their
epoch-2 cost increase. Actual merged measurements instead give cost 0.60342 then 0.59286.
Candidate ranking and all metrics above use the merged outputs, with no substitution of
unmerged results. Both arms retained two epochs and the same 30,104 forwards/3,764 optimizer
steps; the completed training audit checked saved parameter updates and unchanged control
adapters before selection.

## Matched accuracy differences

Differences are full-depth learning minus the matched control at the same epoch. Intervals use
2,000 paired observation-group bootstrap replicates, seed 211. New workflows contain
900 questions in 300 groups; familiar workflows contain 600 questions in 120 groups.

| Epoch | New workflow difference, 95% interval | Familiar workflow difference, 95% interval |
| --- | ---: | ---: |
| 1 | +9.56 pp [+5.89, +13.22] | +0.67 pp [-1.67, +3.00] |
| 2 | +12.56 pp [+8.78, +16.22] | +2.00 pp [-0.0042, +4.1667] |

Both new-workflow intervals exclude zero; both familiar-workflow intervals include zero.
The epoch-2 familiar lower bound is slightly negative, not a demonstrated positive bound.
The full-depth arm has lower uncertainty NLL, Brier and cost at both epochs; probability and
cost comparisons here are point estimates without bootstrap intervals.

Only one training seed was used. These intervals resample observation groups, not workflow
families or training seeds, and have no multiple-comparison correction. Development groups have
been repeatedly inspected. The evidence supports improvement in this matched development
experiment; it does not establish broad business generalization, causally faithful internal
reasoning or an independent final pass. The comparison was computed after fixed candidate
selection and did not choose the checkpoint.

## Evidence and remaining gates

`manifest.json` binds 18 exact aggregate copies: baseline and five candidates, selection
protocol/result, matched comparison, completed training and process records, raw-output
development audit, runtime version preparation, fresh and previous calibration evaluations,
the previous-policy audit and the terminal release status. Model weights, raw logits and original
dataset records are excluded. The development audit and comparison reproduced metrics from
the saved raw outputs and verified source, model and data identities.

The selected weights are
`86ce942f8ef036f2279422a3a2665248b98adcb0effa91bafdf15d9c14c6c83e`; the uncalibrated manifest is
`5b6447b8cea24e6c74845496889c142bbaf6415a34cf91a2293d7b88658bb9ba`.
The selection report SHA-256 is
`18ef8e8beb527ac78933aa8eb6b050e4c8d6880ddc6f41f5a29d093ba29c1bfb`.

The runtime version advanced to `0.3.0a2` only after eligible selection and the raw-output audit.
A wheel was built, and its 38 Python modules were checked against the current sources. This
preparation does not publish a release or substitute for isolated-wheel model inference.

### Calibration fit passed; previous-policy regression failed

Fresh calibration passed on its policy-fitting groups. The identical serialized policy was then
applied to the original, previously inspected policy groups, without refitting.

| Type | Fresh policy-fit coverage | Fresh expected error | Previous-group coverage | Previous expected error | Previous-group pass |
| --- | ---: | ---: | ---: | ---: | --- |
| Choice | 100.00% | 12.33% | 99.85% | 11.10% | Yes |
| Score | 100.00% | 14.98% | 99.73% | 16.95% | No |
| Noul | 90.49% | 15.00%* | 84.68% | 7.64% | Yes |

\* Unrounded fresh noul expected error is 0.14998556535871835, below 0.15.
The fresh score error is 0.14983820821540436. Both fit close to the 15% limit.
The unchanged score policy accepted 367 of 368 previous-group questions and incurred
expected error 0.16954360326228957, above the fixed 15% maximum. This is a performance-gate
failure, not a model-loading or execution fault. The calibrated manifest is
`04fd31900d5fc4d6338cccb0ed4782b9836e93dfa91b055e6a518051b2cf0b35`.

`previous-calibration-policy.json` and `release-status.json` preserve the failure. Final freeze
and final inference did not start. The runtime wheel exists, but its new latency and isolated
package-inference checks were not reached. No workflow HF release package was prepared.

Next work examines score errors and confidence ordering and a more conservative policy-fitting
method on separately declared calibration data. Passing the fresh fit alone cannot replace
the previous-policy regression. Required steps still include that regression, final freeze and
independent final evaluation, retention checks,
resident real-photo HTTP p95 of at most 300 ms, the one-network/zero-generation contract, and
the exact prepared HF package's inference verification. The [original release protocol](../../../configs/workflow-release-v12.json)
and [fresh-calibration provenance](../workspace-calibration/README.md) remain unchanged.
