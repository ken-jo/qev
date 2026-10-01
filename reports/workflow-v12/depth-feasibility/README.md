# Earlier-layer adaptation: execution feasibility only

After every curriculum candidate failed score coverage, a separate preparation measured whether
the same Qwen3.5-2B network can propagate gradients through adapters in all 24 language layers.
The existing checkpoint adapts the last 12. The expanded adapters can be merged into the same
backbone matrices for deployment; this measurement uses unmerged adapters for gradients.
No extra decision network, generated token or runtime prompt was added. This is a memory/gradient
measurement, not a learned model or accuracy result.

The [feasibility declaration](../../../configs/workflow-depth-feasibility-v12.json) fixed four
original training inputs before model execution: the longest serialized request from each of
new workflows, familiar workflows and conditional uncertainty, plus the first training photograph
by record id. This selection does not establish the largest possible token/image shape.
The parent is the highest-ranked but ineligible curriculum epoch-2 checkpoint; it is not an
approved release model. Calibration and final records were not used.

## Measured procedure

New rank-8 adapters start with zero output in the earlier 12 layers. The existing upper adapters,
readout, binding head and probability settings are retained. Initial logits on the first declared
input match the parent exactly (maximum absolute difference 0). Non-reentrant gradient
checkpointing is enabled in the language stack, whose attention dropout is zero. Vision remains
in evaluation mode with frozen weights.

Each declared input ran a forward pass, the existing RLOO+CE objective, backward pass, gradient
clipping and an AdamW step with **all learning rates zero**. This allocates real optimizer state
without learning. The complete adapter/readout state hash was unchanged after every step, and
the original parent files remained unchanged. Every parameter group had finite, present
gradients and a positive aggregate gradient norm.

| Original training input | Input tokens | Peak allocated GPU memory |
|---|---:|---:|
| Complete new workflow | 265 | 4,396.77 MiB |
| Familiar workflow | 693 | 4,625.41 MiB |
| Uncertain workflow | 383 | 4,493.26 MiB |
| Photograph | 375 | 4,494.85 MiB |

Peak reserved memory reached 4,728 MiB. Optimizer groups contain 3,907,584 earlier-adapter
parameters, 3,907,584 later-adapter parameters and 171,072 readout/binding parameters.
These counts exclude the frozen base and frozen condition readout.

The recorded step durations include feasibility bookkeeping and parameter hashing. They are
not inference latency or a controlled training-throughput benchmark. Four successful inputs do
not establish memory bounds for every request, convergence, generalization, calibration or the
300 ms release requirement. No updated model checkpoint was saved.

`manifest.json` binds the copied protocol and measured results; both preserve source identities.
The next training comparison still needs a prospective declaration with matched inputs, update
counts, initialization and selection criteria. The intended factor is whether earlier-layer
adapters receive updates, while both arms retain the same network and original release gates.
