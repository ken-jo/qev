# Recovery training execution repair

[Repair declaration](../../../configs/workflow-backbone-memory-repair-v13.json) preserves the
[original training failure](../backbone-recovery/execution-failure.json). No development,
calibration or final predictions motivated this allocation change.

A training-only forward/backward probe used the eight longest text and eight longest image
requests, including 680 tokens. With a 0.90 process allowance, its maximum allocated GPU memory
was **7,032.36 MiB**; the original 0.85 cap was insufficient. This probe measures allocation
feasibility, not accuracy or long-run stability. Fragmentation was not separately quantified.

The retry uses the identical parent, seed, 9,129 train questions, group order, optimizer schedule,
loss, accumulation and 25%/50%/100% boundaries. Question-local graph references are released;
unused cached memory is cleared after each accumulated batch. Optimizer/scheduler/RNG state is
also saved at existing boundaries for future exact resumption. No model environment or inference
path changes. All original selection/release thresholds remain fixed.

PyTorch's [process cap](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
can itself trigger allocator exhaustion. [Clearing unused cache](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.empty_cache.html)
may reduce fragmentation, but does not create additional physical memory.

The incomplete first run had no saved optimizer state, so the retry starts from the original
parent instead of silently resetting an optimizer midway. First-run partial checkpoints remain
diagnostics and are not extra selection candidates. Reports are added only as stages complete.

At both the 25% and 50% boundaries, all **307 stored tensors** and the complete weight-file
checksums exactly match the corresponding preserved first-run checkpoint. The two
`fraction-*-execution-equivalence.json` files record that comparison. It verifies execution
equivalence through those boundaries; it is not an accuracy or release result.

The retry subsequently reached step 675 / 5,397 training questions, beyond the original
allocation-failure window after step 650. All 26 shared training-log checkpoints have identical
question counts and mean losses. `passed-prior-failure.json` records this execution check;
the full retry then completed **9,129 questions / 1,142 optimizer steps** in 3,106.94 seconds,
with maximum allocated GPU memory **7,050.42 MiB**. `training-complete.json` records completion
and the exact source fingerprints. This establishes successful execution of the declared pass.

Actual merged BF16 development inference completed on the same 5,102 questions for all three
declared checkpoints. Two candidates passed the fixed development screens.

| Training fraction | Legacy text | Legacy image | Development screens |
| --- | ---: | ---: | --- |
| 25% | 90.82% | 97.39% | Text fails the fixed 91% screen; other seven checks pass |
| **50% (selected)** | **91.21%** | **97.39%** | All eight pass |
| 100% | 91.11% | 97.39% | All eight pass |

The 50% candidate also retains familiar/new workflow accuracy (80.00% / 89.33%), uncertainty
NLL (0.75460), distribution Brier (0.13499), and cost at 80% coverage (0.61951) within their
predeclared development limits. These are inspected development results, not fresh final results.
The original ranking first compares the smaller of legacy text/image accuracy. The 50% candidate
won by one correctly classified legacy text development question (934 versus 933 of 1,024),
so the 100% candidate's slightly lower uncertainty NLL does not override that fixed rule.
This small development difference is not a claim of statistically established model superiority.
The selected checkpoint contains 4,533 additional training questions / 567 optimizer steps;
the full 9,129-question run was completed to evaluate every declared candidate.

[Admission](../policy-validation/README.md) independently reconstructed all three candidates'
metrics and selection from saved merged outputs, verified unchanged frozen parameters and
audited data separation. The selected weights have SHA-256
`84b57aeeb987f73416ac0f796150957d1c5beb1ed373733053e46438f77a8fee`.
The unchanged 90% final-release gate, independent policy validation, fresh final evaluation,
runtime and exact-package checks remain mandatory. **No release is approved yet.**
