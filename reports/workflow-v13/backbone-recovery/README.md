# Late-layer logical recovery — first execution stopped

[Prospective design](../../../configs/workflow-backbone-recovery-v13.json) follows the failed
head-only comparison. Parent weights remain the original cohort-policy depth model; an
ineligible head interpolation is not used as initialization.

One pass uses the same 9,129 train questions in a seeded observation-group order. Existing LoRA
parameters in language layers 18–23 and the existing decision readout can learn; earlier
adapters, base weights, vision and condition readout are frozen. Backbone/head learning rates
are 1e-5 / 1e-6, with accumulation 8 and the declared warmup/cosine schedule. Training uses
frozen-temperature soft cross entropy and 4× temperature-2 KL to current merged outputs on
workflow replay. This repair stage uses supervised replay and distillation; it is not a new
reinforcement-learning algorithm. Earlier parent adaptation included RLOO plus cross entropy.

The 25%, 50% and 100% checkpoints are fixed in advance. All three must be measured with actual
merged BF16 GPU inference on all 5,102 dev questions. The same 91% legacy development screens,
workflow/uncertainty retention and ranking remain fixed. No calibration/final outputs select
weights. Existing inference modules, prompts, one-forward contract and zero output tokens stay
unchanged. Training-only activation checkpointing applies to the six trainable language layers.

`protocol.json` and `parameter-plan.json` bind the exact input/source hashes, order, trainable
parameter names and planned steps. Training completion and development reports appear only
after their stages finish. No improved accuracy or release pass is claimed yet.

The first execution stopped with CUDA out of memory after saving the 25% and 50%
checkpoints. The 0.85 allocator allowance was insufficient for the longest training inputs.
`execution-failure.json` preserves the failure and last progress; neither partial checkpoint
was evaluated on development, calibration or final populations. A
[separate memory repair](../backbone-memory-repair/README.md) restarts the exact declared
experiment from the same parent, retaining all three original selection boundaries.
