# Completed head-only recovery — no eligible candidate

[Prospective design](../../../configs/workflow-readout-recovery-v13.json). The current 24-layer
adaptation, vision, condition readout and prompts stayed fixed. Only the existing option readout
and condition-binding head were mixed or trained. No inference module was added.

The cache contains 9,129 train and 5,102 dev questions. Train groups do not intersect any
calibration/dev/final group across the two declared source corpora. Exact train/dev requests and
images do not overlap. Features use the actual merged current model; no calibration or final
features were extracted. The original train data, including logical counterfactual pairs, supply
the repair examples; original failed final examples were not optimized.

| Candidate | Legacy text dev | Legacy image dev | Eligible |
| --- | ---: | ---: | --- |
| Current head, reference | 87.01% | 96.87% | No |
| 25% Foundation head | 88.18% | 97.13% | No |
| 50% Foundation head | 88.87% | 97.00% | No |
| 75% Foundation head | 89.26% | 97.13% | No |
| 100% Foundation head on current backbone | 89.84% | 97.26% | No |
| Replay epoch 1 | 89.65% | 97.13% | No |
| Replay epoch 2 | 89.26% | 97.13% | No |
| Replay epoch 4 | 87.89% | 96.87% | No |
| Replay epoch 8 | 87.70% | 96.87% | No |
| Replay epoch 12 | 87.89% | 96.74% | No |
| Replay epoch 20 | 87.79% | 96.21% | No |

The predeclared development screen was 91% per legacy modality, along with retained workflow,
public-data accuracy and probability quality. The original release threshold remains 90%.
The highest legacy score among these ineligible candidates comes from the complete Foundation
head; its familiar-workflow dev accuracy falls to 63.17% from 80.67%. It is not promoted.
Replay completed all 20 epochs / 1,440 CPU optimizer steps. Earlier declared checkpoints and
all negative results are retained. No policy fitting, final inference or release occurred.

Head interpolation is inspired by [parameter averaging](https://arxiv.org/abs/2203.05482),
using only aligned readout parameters. This is not a reproduction of the full model-soups
experiment, and it did not satisfy the declared recovery screens here.
