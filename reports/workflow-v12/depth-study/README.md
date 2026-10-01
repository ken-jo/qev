# Matched depth adaptation study

The [prospective declaration](../../../configs/workflow-depth-study-v12.json) fixes the
learning and selection design after the previous curriculum study failed the score screen.
Both arms have completed their two declared epochs and unmerged development evaluations.
The completed-study training audit passed. The subsequent [merged comparison](../depth-selection/README.md)
selected full-depth epoch 2 after evaluating all five candidates. This page preserves the
unmerged training diagnostics. **Independent final performance and release acceptance remain
separate requirements.**

| Item | Fixed design |
| --- | --- |
| Parent | Highest-ranked but ineligible curriculum epoch 2 |
| Arms | `full_depth` and `late_control` |
| Initialization | Same rank-8 adapters on all 24 layers; new earlier-layer B matrices are zero |
| Difference | Earlier 12 adapters have learning rate `3e-5` versus `0`; later/readout schedules match |
| Work per arm | 2 epochs; 15,052 identical queries per epoch; 3,764 optimizer steps |
| Questions | 11,212 original questions plus one fact and one rule question on each of 1,920 training states |
| Seed | 211; this single seed does not establish robustness |
| Memory | Earlier-12 activation checkpointing through 693 tokens; all-24 for longer inputs |
| Selection | Parent and four learned checkpoints, using original merged development gates |
| Release | Priorities 1 and 2, retention, previous-policy regression and 300 ms HTTP p95 remain mandatory |

Both arms compute gradients and AdamW state for all groups. The zero-rate control must retain
its earlier parameter fingerprints, and its saved earlier B matrices must stay zero. All trained
checkpoints retain the parent prompt, frozen vision stack, decision heads and runtime contract.
Adapters merge into existing backbone matrices; no inference module or answer generation is added.

`training-plan-audit.json` enumerates the fixed schedule and verifies exact original-query
coverage, state/group identity and one fact/rule extra per training state. The durable
`scripts/depth_training_checks.py` independently repeats these checks after training and verifies
the saved checkpoints and matched initialization before merged selection. Its completed-study
audit passed and is preserved in [merged-selection-protocol.json](merged-selection-protocol.json).
All 37 training-audit evidence hashes were independently rechecked against local files.

`cached-development-audit.json` recalculates both Foundation and parent metrics from their
existing raw logits. It makes no new GPU prediction and does not inspect calibration or final
model performance. Startup JSON files here are exact copies of local evidence; `manifest.json`
binds them. Raw data, logits, optimizer states and weights are not included.

## Interim epoch records

The [first](full_depth-epoch-1.json) and [second](full_depth-epoch-2.json) full-depth epochs are
compared below with the [experimental parent](../skill-curriculum/skill_curriculum-epoch-2.json).
All three model columns use unmerged adapters on the same development questions. The completed
merged measurements are reported separately in the linked comparison.

| Development metric | Parent | Full-depth epoch 1 | Full-depth epoch 2 |
| --- | ---: | ---: | ---: |
| New workflow accuracy | 76.33% | 85.33% | 90.00% |
| Familiar workflow accuracy | 80.00% | 77.17% | 80.83% |
| Photo accuracy | 96.67% | 96.67% | 97.00% |
| Sentence-relation accuracy | 86.89% | 89.78% | 89.11% |
| Intent accuracy | 83.98% | 84.63% | 84.42% |
| Uncertainty NLL | 0.87589 | 0.83309 | 0.77423 |
| Uncertainty Brier against soft targets | 0.21515 | 0.17176 | 0.13642 |
| Uncertainty expected cost at 80% coverage | 0.86918 | 0.59051 | 0.62055 |
| Score coverage under 15% expected error | 29.69% | 64.38% | 93.91% |
| Score expected error at 60% coverage | 21.53% | 14.52% | 10.59% |

Both epochs exceed the 60% score coverage diagnostic; epoch 2 choice/noul coverage is
100.00%/100.00%. The first-epoch familiar workflow regression recovers in epoch 2. NLL and
Brier improve, while uncertain-input cost at 80% coverage increases compared with epoch 1;
it remains below the parent's cost. These are
descriptive interim results from one seed and repeatedly inspected development groups. The
matched control has completed both epochs, compared below. These diagnostics do not grant
release eligibility; independent policy calibration and all later release gates remain necessary.

Epoch 1 binds weights `51fd8826f281d627d5390e1d073d1c31f10d5cec44cb6d8609987480fde34ace`
and manifest `be5794e0cfcd4e3f12fd571d44f9e47da1520ebdfb7363214c2cf2cc05917087`.
It completed 15,052 training forwards and 1,882 optimizer steps. The saved record confirms
earlier adapters changed and peak allocated memory was 6,638 MiB. Skill-query diagnostics are
included for explanation, not candidate selection.

Epoch 2 binds weights `86ce942f8ef036f2279422a3a2665248b98adcb0effa91bafdf15d9c14c6c83e`
and manifest `5b6447b8cea24e6c74845496889c142bbaf6415a34cf91a2293d7b88658bb9ba`.
The full-depth arm completed 30,104 training forwards and 3,764 optimizer steps; peak allocated
memory remained 6,638 MiB. `full-depth-complete.json` records completion of this arm only.

`late-control-protocol.json`, `late-control-initialization.json` and
`late-control-first-gradient.json` preserve the control startup. Trainable parameter and group
fingerprints, parameter counts, expansion layout, fixed configuration and first gradient norms
match the full-depth arm. The declared earlier-layer learning-rate multiplier is 0 in the control
and 1 in the full-depth arm. Both control checkpoints' saved zero-update checks and the
complete-study training audit passed. The earlier A and B tensors are exactly identical across
the two control checkpoints; all 75 earlier B matrices remain zero.

### First matched epoch, unmerged

The [first control checkpoint](late_control-epoch-1.json) uses the same 15,052 training queries,
1,882 optimizer steps and initialization as full-depth epoch 1. Both use seed 211 and the same
development groups. This comparison reports unmerged point estimates; paired merged statistics
are in the separate completed comparison.

| Development metric | Full-depth epoch 1 | Control epoch 1 |
| --- | ---: | ---: |
| New workflow accuracy | 85.33% | 75.89% |
| Familiar workflow accuracy | 77.17% | 77.17% |
| Photo accuracy | 96.67% | 96.67% |
| Sentence-relation accuracy | 89.78% | 86.89% |
| Intent accuracy | 84.63% | 83.77% |
| Uncertainty NLL | 0.83309 | 0.85451 |
| Uncertainty Brier against soft targets | 0.17176 | 0.20243 |
| Uncertainty expected cost at 80% coverage | 0.59051 | 0.77513 |
| Score coverage under 15% expected error | 64.38% | 24.22% |
| Score expected error at 60% coverage | 14.52% | 22.20% |

The new-workflow point difference is +9.44 percentage points for full-depth epoch 1. Familiar
workflow accuracy is identical, and remains below the parent's 80.00% in this first epoch.
Control choice/noul coverage under 15% error is 98.66%/74.14%; score coverage fails the 60%
screen. This result does not change the declared two epochs per arm or five-candidate selection.
A single seed and repeatedly inspected development groups cannot establish a general depth
advantage, independent final performance or release eligibility.

Control epoch 1 binds weights
`d1f227739117be1b93a435387455321eb4ddd4b6e7e3388a1c0e183a9d72e66c` and manifest
`6a5b5d5acf7b713d473854f7bc9e24700e18d7fdf344cb7e70c0a99d97c3f3f1`.
CPU inspection of the actual saved tensors confirmed all 75 earlier B matrices are exactly
zero, all values are finite, and the frozen condition readout and inference contract are
preserved. Its recorded earlier parameter-group fingerprint equals the initialization. It stores
7,992,384 parameters and reached peak allocated memory of 6,638 MiB. The source fingerprints
still match the declared training protocol. The subsequent complete-study audit also passed.

### Second matched epoch, unmerged

The [second control checkpoint](late_control-epoch-2.json) completed the same 30,104 training
forwards and 3,764 optimizer steps as full-depth epoch 2. Both arms finished their original
budgets; neither was stopped early or given extra training after inspecting these results.

| Development metric | Full-depth epoch 2 | Control epoch 2 |
| --- | ---: | ---: |
| New workflow accuracy | 90.00% | 77.00% |
| Familiar workflow accuracy | 80.83% | 78.50% |
| Photo accuracy | 97.00% | 96.50% |
| Sentence-relation accuracy | 89.11% | 87.33% |
| Intent accuracy | 84.42% | 84.20% |
| Uncertainty NLL | 0.77423 | 0.87382 |
| Uncertainty Brier against soft targets | 0.13642 | 0.20840 |
| Uncertainty expected cost at 80% coverage | 0.62055 | 0.84284 |
| Score coverage under 15% expected error | 93.91% | 26.09% |
| Score expected error at 60% coverage | 10.59% | 20.28% |

The new-workflow point difference is +13.00 percentage points. Control choice/noul coverage
under 15% error is 99.47%/82.76%; its score coverage still fails the 60% screen. These unmerged,
single-seed development results do not establish independent generalization or release
eligibility. The declared merged comparison retains the parent and all four trained checkpoints.

Control epoch 2 binds weights
`612a5ebd4b07fbb5c4396ad002ab24b8653be1615cd918bc2997e672fcf0a832` and manifest
`f7ec0cd11903e7f80d45246cf3e5862e5348f5fa0a1d00efa324bc8fc30356de`.
CPU inspection verified finite values and the expected 7,992,384 stored parameters in all four
checkpoints, unchanged frozen condition readouts and inference contracts, and identical earlier
control tensors across epochs. `late-control-complete.json` and `study-complete.json` preserve
the actual completion records. Peak allocated training memory was 6,638 MiB for both arms.

`scripts/export_depth_progress.py --wait` observes completed checkpoint manifests and preserves
their training metadata, initialization fingerprints, unmerged development metrics, probability
diagnostics and skill diagnostics. Each aggregate binds the actual checkpoint weights, manifest,
training protocol, fixed declaration and exporter. Existing public evidence hashes are checked
before adding records. The observer uses no GPU and does not alter learning or selection.

Until a completed checkpoint exists, the observer reports a pending state without creating a
result. Later `full_depth-epoch-N.json` and `late_control-epoch-N.json` files are explicitly
`interim_unmerged_development`; merged selection, independent calibration and final acceptance
remain necessary. Reference-arm epoch records do not perform a control comparison. The startup
evidence above supplies this check; completed control-epoch reports will repeat it.

The prepared continuation is training → merged development → raw-output audit → fresh
calibration → unchanged old-policy regression → final freeze/evaluation → regressions and
HTTP/one-network verification → acceptance → HF package preparation and isolated-wheel inference.
Every unmet mandatory gate stops advancement. Training, merged selection and the raw-output
audit are complete. The subsequent [previous-policy regression failed](../depth-selection/README.md#calibration-fit-passed-previous-policy-regression-failed)
for score questions, stopping the release evaluation before final inference. Actual HF publication
still requires a separately configured account and target.

## Packaging input audit

`hf-evidence-inventory.json` records the startup packaging-input audit: 14 included source/metadata
entries and two external inputs (the original records and feature cache). Every input hash was
verified in that snapshot. Actual packaging requires the eventual acceptance inventory bound to
the evaluated sources and checkpoint. It records excluded input fingerprints and maps included
evidence. The startup inspection created no release package or model-performance result.
