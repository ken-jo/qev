# Merged deployment development comparison

All candidates use the same 3,312 development questions and actual merged BF16 deployment.
This is development selection; calibration and final results are not included.

| Candidate | Known workflow accuracy | New workflow accuracy | Uncertainty NLL | Error cost at 80% coverage | Development eligibility |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen Foundation | 50.33% | 44.67% | 1.072679 | 1.999765 | baseline |
| RLOO readout | 68.83% | 53.00% | 0.975207 | 1.414098 | true |
| Low LR, half epoch | 70.83% | 52.11% | 0.986070 | 1.501801 | true |
| Low LR, full epoch | 72.67% | 55.00% | 0.978178 | 1.436804 | true |
| High LR, half epoch | 71.83% | 54.33% | 0.986922 | 1.351836 | true |
| High LR, full epoch | 74.33% | 58.78% | 0.973591 | 1.297562 | true |

Selected: **high learning rate, full epoch** under the previously fixed development ranking.
The weights and original manifest are bound in `selection.json`. All five candidates satisfy
the development eligibility checks; this does not imply passing calibration or final release gates.

| Retention domain | Foundation | Selected | Change |
| --- | ---: | ---: | ---: |
| photo_guard | 95.83% | 96.17% | +0.33 pp |
| text_nli | 84.44% | 86.44% | +2.00 pp |
| text_intent | 83.98% | 83.98% | +0.00 pp |

Known workflow results are development teacher-label agreement, not the official held-out
workflow regression. New workflows are procedurally constructed family-transfer tasks.
Uncertainty targets follow the declared prior and sensor assumptions, not measured real-world
event frequencies. Every original report is copied byte-for-byte; `export.json` records hashes.
