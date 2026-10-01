# Condition-role diagnosis after failed merged selection

This exploratory measurement ran only after the previous GPU evaluation had stopped at its
development gate. It uses 1,200 eligible development requests and a fixed training-only decoder
from the condition-supervision arm, without fitting weights, policy or calibration.

| Checkpoint | Condition probability MSE | Broadcasting the four-prediction mean: MSE | Observed condition accuracy | Broadcast accuracy |
|---|---:|---:|---:|---:|
| Parent | 0.20990 | 0.20420 | 62.20% | 64.00% |
| Condition supervision, epoch 2 | 0.18020 | 0.20631 | 69.64% | 62.77% |
| Matched continuation, epoch 2 | 0.18885 | 0.20380 | 68.80% | 63.21% |

The four supervised positions contain some distinct information: the actual assignment beats
broadcasting and random role permutations for the supervised model. The hypothesis that all four
positions simply encode the same condition is not supported. Its observed A/B/C/D accuracies are
60.18%, 65.72%, 70.85% and 82.09%; all four conditions are numeric-threshold comparisons in the
source data. This uneven decoding result motivates directly examining prerequisite skill queries.

The fixed decoder was learned only with the supervised model. Applying it to the parent/control
is not a fair capacity comparison using independently refitted probes. Similarity between hidden
states does not prove collapse, and decoded accuracy does not prove causal use in the final
decision. Best-assignment MSE in the JSON reports uses true targets and is only a reference.

`protocol.json`, `results.json` and `complete.json` bind model, source and data hashes. The raw
hidden-state tensors stay local. No inference prompt, runtime module, model weight, acceptance
threshold, fresh calibration or final prediction was changed.
