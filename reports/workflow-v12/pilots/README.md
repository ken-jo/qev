# Development-only improvement pilots

These pilots were declared before their training runs. They reuse the immutable Foundation
training/development representations and matched RLOO controls at seeds 131, 137 and 139.
Each method is ranked by its median development selection tuple; seed 131 is the representative.
Neither calibration nor final data is used, and no pilot is silently added to the five candidates
of the already frozen primary experiment. A different candidate needs a separately declared
complete evaluation cycle with the same mandatory release thresholds.
The separate active backbone continuation study also keeps its original candidate list.
The confidence pilot below uses its own declared cross-fitting and critic seeds.

`artifact-contract.json` confirms that the first two representative pilot checkpoints keep the baseline's
tensor names, shapes, 4,084,800 stored parameters and architecture settings. The auxiliary decoder
is absent from model weights. This static check does not replace runtime latency or forward-count
validation for any eventual release candidate.

## Results for the representative seed

These are unmerged cached-feature development results, not deployed/final measurements.

| Readout | Known workflow accuracy | New workflow accuracy | Uncertainty NLL | Cost at 80% coverage |
| --- | ---: | ---: | ---: | ---: |
| Matched RLOO control | 68.83% | 53.00% | 0.975207 | 1.414098 |
| Typed consistency, weight 0.25 | 68.50% | 56.56% | 0.973090 | 1.644044 |
| Condition supervision, weight 0.25 | 68.33% | 56.11% | 0.973428 | 1.639418 |

Both pilots improve new-workflow accuracy over the matched readout control, but their expected
error cost at the same coverage worsens. They are not uniform improvements and do not establish
that probability/abstention release requirements can be met. All seed outcomes and the stronger
consistency-weight result remain in the JSON reports.

### Equivalent questions should agree

`workflow_consistency.py` adds a symmetric Jensen-Shannon loss between choice and ordinal score
distributions over the same outcomes, and their binary projection versus the noul distribution.
Only complete training questions with identical state and checked semantic targets are coupled.
Different official-workflow questions and uncertain views are not coupled. The loss adds no
inference module or calls; outcome mappings exist only in training annotations.

### Teach primitive conditions as well as final answers

`workflow_facts.py` derives A–D condition probabilities from the explicitly stated numeric
thresholds, priors and independent-sensor assumptions in training/development requests. The
conditional targets match the preserved generator audit. A four-output auxiliary decoder
supervises the existing 64-dimensional readout projection; the decoder is discarded from the
deployment checkpoint. No parsed truth values are provided to model inference.

The selected auxiliary decoder's observed-condition accuracy is 63.31% on training and 53.99%
on development, versus 60.36% and 51.55% for training-marginal predictions. Development condition
probability MSE is 0.217211 (constant 0.5: 0.227218; training marginals: 0.233047). Thus the probe
extracts some condition information but remains weak on novel families. This diagnoses this
particular frozen representation/readout setup, not an upper bound on the Qwen model.

## Remaining weakness and next decision

At 60% development answer coverage, score expected error remains about 37% and noul about 22%
for both pilots. Their maximum coverage under 15% expected error is still below the required
60%. No deployed policy was fitted by this diagnostic.

The primary backbone experiment subsequently failed independent score/noul calibration. A
separately declared matched study now passes primitive-condition supervision into the backbone,
preserving training-only replay and every release gate. Its remaining epochs, matched control
and merged development evaluation are required before judging this change.

## Target-semantics pilot

After a [14,524-question mapping audit](../training-target-audit.json) found no target-position
errors, a [separate protocol](../../../configs/workflow-mode-target-pilot-v12.json) compared
two blends of teacher probabilities and recorded hard gold on the 4,200 teacher-distribution
training questions. The conditional-probability and retention targets were unchanged. All other
settings matched the three existing RLOO controls: 25 epochs, the same seeds, parent, learning
rates, group batches, source shares and replay. `teacher-mode/` preserves the complete protocol,
all nine control/experimental results and the development-only comparison.

| Hard-gold weight, seed 131 | Known workflow accuracy | New workflow accuracy | Uncertainty NLL | Cost at 80% coverage | Score error at 60% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% (matched control) | 68.83% | 53.00% | 0.975207 | 1.414098 | 37.39% |
| 50% | 69.83% | 56.44% | 0.972661 | 1.729524 | 36.05% |
| 100% | 71.83% | 52.89% | 0.984249 | 1.891136 | 35.43% |

The original median selection rule ranks 100% first, but **none of the six new runs reaches the
mandatory score coverage requirement**. The best score threshold coverage under 15% error is
only 6.09%, at seed 139 with 100% hard gold. Most runs worsen uncertain-error cost versus their
matched seed. The original rule's ranking is not release approval, and the representative seed
is not replaced with the most favorable seed. Target hardening alone is not supported as a
solution. These results motivate separating prediction quality and confidence ordering in the
pending backbone comparison; no pilot changes the deployed model or consumes final groups.

## Learned confidence from withheld training families

The [confidence protocol](../../../configs/workflow-reliability-pilot-v12.json) declared a
separate CPU experiment before fitting. Three readout folds each withheld two complete procedural
training families; related views stayed together. Their out-of-fold predictions supplied targets
for a 65,857-parameter confidence head using the same hidden state and eight probability/type
summaries. Three critic seeds ran for 40 fixed epochs. Decisions and class distributions stayed
unchanged, and no threshold was fitted for deployment. `reliability/` preserves the protocol,
all seed results and comparison.

| Confidence ordering | Choice coverage | Score coverage | Noul coverage | Uncertainty cost at 80% coverage |
| --- | ---: | ---: | ---: | ---: |
| Original maximum probability | 83.70% | 0.00% | 6.90% | 1.414098 |
| Learned, seed 173 (declared representative) | 84.42% | 0.00% | 46.21% | 1.434052 |
| Learned, seed 179 | 84.66% | 16.41% | 40.52% | 1.420422 |
| Learned, seed 181 | 84.70% | 10.47% | 43.62% | 1.489966 |
| Median of the three learned runs | 84.66% | 10.47% | 43.62% | 1.434052 |

Coverage is the maximum development confidence-threshold coverage under 15% expected error,
with at least 20 accepted questions and ties kept together. The screen required at least 60%
for **every** type and no increase in uncertain-error cost. It failed. Noul ordering improved,
but this result does not support adding this head to the release model.

This pilot uses the older frozen Foundation representations and RLOO readout, not the active
backbone candidate. Cross-fitting excludes labels from the new readout fits; the frozen parent
had already adapted on some legacy training domains. It is not backbone cross-fitting, and its
failure does not establish that all learned-confidence methods fail. Calibration/final records
were not used for fitting or measurement, and no inference module was integrated.

[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) and
[ConfidNet](https://arxiv.org/abs/1910.04851) motivate learned rejection/confidence. This experiment
predicts out-of-fold expected correctness; it neither reproduces those algorithms nor claims
a new algorithm.
