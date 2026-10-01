# Oracle risk and remaining model error on the consumed v12 final

The newer, separately scoped [recovery final diagnostic](recovery-final.md) records remaining
uncertainty error for the selected v13 weights. The two final populations differ.

This is a **post-hoc diagnostic of the previous v12 cohort-policy model**, whose release failed.
It does not evaluate the recovery model and uses no fresh v13 validation/final observations.
[Reproduction script](../../../scripts/diagnose_consumed_final_uncertainty.py) verifies the
previously recorded model, data and prediction hashes and reconstructs every reported metric.

The synthetic observations have exact conditional target distributions. For each request,
the minimum expected classification error is `1 - max(target probabilities)`. Subtracting it
from the model's expected error gives decision regret on that **same request**. The bound uses
the authored latent/sensor assumptions, not real-world event frequencies or additional evidence.

| Identical request set | Questions | Cohort model expected error | Oracle minimum | Additional decision error |
| --- | ---: | ---: | ---: | ---: |
| All uncertain requests | 480 | 37.91% | 22.14% | 15.76 points |
| Requests accepted by the cohort policy | 256 | 33.45% | 19.60% | 13.85 points |
| Cohort confidence top 80% | 384 | 33.73% | 19.95% | 13.78 points |

For the **same 256 accepted requests**, Foundation's expected error was 63.20%, compared with
33.45% for the cohort model. This fixes membership when comparing decisions; comparing each
model's own accepted sets would compare different requests. Even oracle decisions on this
particular accepted set cannot achieve 15% error. That does not change the declared calibration
requirements, which apply to a different mixture and population. Achieving lower uncertainty
subset error may require better decisions, different abstention membership/coverage, or more
observed evidence. The full 13.85-point gap is not a promised achievable gain for this model.

| Uncertainty condition, all 160 requests each | Model expected error | Oracle minimum | Additional decision error |
| --- | ---: | ---: | ---: |
| Missing evidence | 43.22% | 29.55% | 13.67 points |
| Conflicting evidence | 44.06% | 28.27% | 15.79 points |
| Shifted prior | 26.44% | 8.61% | **17.83 points** |

The prior-shift condition has the lowest raw error and the greatest excess decision error.
Raw error alone would therefore obscure an important remaining improvement area. This is a
training-design hypothesis for future research, not permission to optimize these final examples
or change the ongoing experiment's selection criteria.

## Probabilities and costs

Across all 480 requests, conditional entropy is 0.51193. Foundation NLL 1.02859 contains 0.51666
excess NLL; cohort NLL 0.81672 contains 0.30478 excess NLL. The excess equals conditional KL
up to the evaluator's probability-clipping epsilon. The reported Brier metric is squared error
against the conditional distribution, with oracle minimum **zero**; it is not event-label Brier
with an added irreducible noise term.

On the cohort's fixed 256 accepted requests, expected 0/1/5 cost is 0.85561 and the oracle
minimum cost is 0.32885. The action minimizing cost may differ from the one minimizing error.
No oracle action, target-derived confidence or new cost policy was inserted into inference.

The JSON includes all-request, fixed accepted-set, each model's own accepted-set, matched-count
and per-condition statistics for both models. It contains aggregate numbers and fingerprints;
no original records or images are redistributed. These numbers are exact averages for the
inspected synthetic population and do not provide a deployment-risk guarantee.
**Every original release threshold remains unchanged.**
