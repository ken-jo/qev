# Remaining uncertainty error in the completed recovery final

This post-hoc diagnostic uses only the **completed, frozen v13 final predictions**. It performs
no new model inference and changes no weights, fitting rule, selection or release requirement.
[Source](../../../scripts/diagnose_recovery_final_uncertainty.py) ·
[Aggregate output](recovery-final-uncertainty.json) ·
[Final evaluation](../final-evaluation/README.md).

## Same-request error decomposition

The authored latent/sensor model supplies exact conditional target probabilities. Under those
assumptions, the minimum possible classification error is `1 - max(target probabilities)`.
The difference from model expected error is additional decision error on the same requests.
These bounds do not measure unknown real-world event frequencies or guarantee achievable gains.

| Recovery request set | Questions | Model expected error | Oracle minimum | Additional decision error |
| --- | ---: | ---: | ---: | ---: |
| All uncertain requests | 480 | 44.54% | 22.28% | 22.26 points |
| Requests accepted by the frozen recovery policy | 252 | 45.57% | 21.89% | 23.68 points |

On the **same 252 accepted requests**, Foundation's expected error is 59.43%, compared with
45.57% for recovery. The model improves relative to Foundation but still makes substantial
avoidable errors. Even oracle decisions on this exact accepted set cannot achieve 15% error.
The declared 15% fitting/validation limit applies to different mixed populations and is not
a final uncertainty-subset risk guarantee.

Within the 480 uncertain requests, actual abstention accepts 52.50%. Its accepted-answer
expected error (45.57%) is higher than the always-answer average (44.54%). The corresponding
expected 0/1/5 cost is lower, 0.95691 versus 1.05757. This describes this policy's selected
mixture; it does **not** establish reliable ranking by probability of being correct on new
uncertain tasks. The matched-80%-coverage comparison against Foundation remains a separate,
predeclared release metric.

## Conditions and probability quality

| Condition, 160 requests each | Model expected error | Oracle minimum | Additional decision error |
| --- | ---: | ---: | ---: |
| Missing evidence | 46.66% | 30.47% | 16.19 points |
| Conflicting evidence | 44.47% | 26.82% | 17.65 points |
| Shifted prior | 42.50% | 9.54% | **32.96 points** |

Across all uncertain requests, NLL is 0.89909 and conditional entropy is 0.51450, leaving
0.38459 excess NLL. On accepted uncertain requests, NLL is 1.01097 and excess NLL is 0.49445.
Squared error against the conditional distribution has theoretical minimum zero.

The predeclared paired analysis improves aggregate NLL from 1.00329 to 0.89909, with a 95%
delta interval of −0.15039 to −0.05638. **Shifted-prior NLL alone is not conclusively improved**:
its delta interval is −0.22660 to +0.01562. Those intervals are observation-group bootstraps
without adjustment for multiple comparisons. The aggregate gate and condition-level evidence
have different scopes and must not be interchanged.

Overall final 15-bin ECE also worsens, from 6.80% to 12.58%. Lower aggregate NLL, Brier and
matched-coverage cost therefore do not establish universally better confidence or abstention.
These limitations belong in the roadmap and HF documentation alongside the passing metrics.

The earlier v12 accepted-error result (33.45%) used different final families, observations and
accepted membership. Comparing it directly with 45.57% is not a paired model-regression result.
Both reports preserve their own exact inputs and model identities. Future improvement must use
training/development evidence and reserve new final observations; these examples cannot become
an independent final benchmark again.
