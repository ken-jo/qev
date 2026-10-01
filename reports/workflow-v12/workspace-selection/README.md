# Matched backbone study: merged development results

**No candidate qualified. No fresh calibration or final prediction was evaluated.**

The separately declared study compared condition supervision at four existing internal positions
with ordinary continuation, using the same parent, seed, order and two training epochs.
All five candidates were evaluated with their adapters merged into BF16 base weights, as deployed.
The Foundation baseline used exactly the same 3,312 development requests.

| Merged model | New workflow accuracy | Known workflow accuracy | Uncertainty NLL | Cost at 80% coverage | Score coverage at <=15% error |
|---|---:|---:|---:|---:|---:|
| Foundation | 44.67% | 50.33% | 1.07268 | 1.99976 | 0.00% |
| Parent | 58.78% | 74.33% | 0.97359 | 1.29756 | 3.44% |
| Condition supervision, epoch 1 | 69.89% | 73.50% | 0.96071 | 1.56707 | 5.31% |
| Matched continuation, epoch 1 | 69.33% | 73.83% | 0.96008 | 1.58082 | 6.88% |
| Condition supervision, epoch 2 | 73.11% | 77.17% | 0.96240 | 1.30693 | 20.63% |
| Matched continuation, epoch 2 | 72.56% | 77.33% | 0.96455 | 1.32169 | 22.81% |

Every candidate fails the required 60% score coverage under 15% expected error. The `selected`
entry in `selection.json` is the ranking result among failed candidates; **`eligible: false`**
prevents calibration or release. Choice and noul screening success cannot compensate for score.
Retention and improvements over Foundation do not constitute full release acceptance.

## Is condition supervision better than the matched control?

The epoch-2 novel-workflow difference is +0.56 percentage points. A paired group bootstrap
(300 observation groups, 2,000 replicates, fixed seed 191) gives a descriptive 95% interval
of -0.67 to +1.78 percentage points. Both epochs' intervals contain zero. The larger improvement
over the parent therefore must not be attributed specifically to the auxiliary supervision.
Adapter updates differ, but different weights do not demonstrate faithful internal reasoning.

These intervals concern repeatedly inspected development groups. They do not resample training
seeds or workflow families, are not adjusted for multiple comparisons, and are not independent
final evidence. The study uses only one matched training seed.

## Confidence ordering diagnosis

Four fixed orderings were applied to the same saved outputs: maximum probability, probability
margin, top-two conditional probability, and one minus normalized entropy. None passed score
coverage for any checkpoint. The highest score coverage among these descriptive combinations
was 23.75% (condition-supervision epoch 2 with entropy ordering), still below 60%.

The alternative scores are ranking statistics, not calibrated correctness probabilities. They
change neither class predictions nor distributions, NLL or distribution Brier. No alternative
policy was fitted or adopted; the declared selection stayed unchanged. Any later policy study
requires its own declaration, fresh calibration and all unchanged release gates.

## Evidence

- `selection.json`: all candidates, original eligibility, per-type risk and diagnostic breakdowns.
- `merged-baseline-dev.json`, `merged-dev-0.json` through `merged-dev-4.json`: full aggregate metrics.
- `matched-development-comparison.json`: paired outcomes, bootstrap intervals and parameter deltas.
- `confidence-ordering-diagnostic.json`: all six checkpoints and all four fixed orderings.
- `failure.json`: evaluation stopped at development with `final_predictions_started: false`.
- `manifest.json`: exact SHA-256 hashes of the copied evidence files.

Raw data, images, per-request hidden states and model weights are excluded from this source report.
