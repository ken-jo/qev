# Training lineage and reproduction

The frozen checkpoint was developed through starter typed heads, synthetic policy training,
Foundation photo/text transfer, workflow condition/binding studies and final recovery.
The original experiment names remain in files to preserve source and weight provenance.

| Stage | Main inputs | Outcome |
| --- | --- | --- |
| Starter and dynamic policy | Starter v2, policy v3/v6/v8 | Typed option decisions and language adaptation |
| Foundation v11 | TrashNet, Beans, SNLI, BANKING77, authored policy/uncertainty | Common text/photo transfer |
| Workflow v12 | Authored rule workflows, typed-decisions train and replay | Condition slots/binding; several failed release candidates |
| Cohort calibration | Recorded v12b-f/cohort populations | Calibrated policy studies with separate observation groups |
| Recovery v13 | Workflow v12 + policy v8 train/dev | Selected late-layer half-pass, new policy validation and fresh final |

The latest recovery used 9,129 declared training questions and 5,102 development questions.
Three prospective checkpoints were evaluated. The selected half-pass consumed 4,533
questions / 567 optimizer steps, with seed 271, gradient accumulation 8, adapter LR 1e-5
and readout LR 1e-6. Only language adapter layers 18-23 and the readout/binding heads
updated. Supervised soft cross-entropy and parent-distribution KL replay were used.
See [the original design](../configs/workflow-backbone-recovery-v13.json).

An earlier head-stage comparison evaluated CE, direct proper-score objectives, differentiable
noise, RLOO and cross-question probability consistency under a shared budget. CE was
selected. The evidence does not establish superiority over reinforcement learning generally.

## What is reproducible from the release

- Inference with the exact adaptation tensors and calibrated manifest.
- Schema, calibration, scoring and serving code, with a branded facade over the preserved runtime.
- Licensed corpus snapshots with original IDs, splits, image bytes, source revisions and hashes.
- Training/evaluation/generator scripts, fixed protocol configurations and numerical reports.

Historical scripts use their recorded `data/`, `runs/` and `checkpoints/` paths. Extract a
corpus ZIP into `data/` to restore its expected layout. Eligible record bytes are preserved;
CIFAR evaluation rows/images are omitted from redistribution. Download those directly from
the cited upstream source and use the source preparation scripts for an exact full evaluation.

Full end-to-end training from an empty machine has **not** been re-run for this publication.
Intermediate teacher/checkpoint/cache inputs are recorded by hash but are not all distributed.
Do not describe the current upload as a turnkey, byte-identical full retraining pipeline.
The trained release is usable independently of those intermediate research artifacts.

## Mix-training and early-exit recipes

Two research recipes continue from a released option checkpoint. They need a CUDA GPU,
are not run in CI and have no published results yet.

`scripts/train_option_mix.py` merges the released LoRA, installs fresh rank-8 LoRA on every
text layer and trains it together with the readout, condition readout and binding heads.
`--pixels 262144,655360,1638400` draws a per-image budget for each request
(resolution-elastic training); multi-image requests share `--total-pixels`. `--permute`
shuffles option order on every visit so the readout binds to option content rather than
letter position; `--fixed-order-sources` lists sources whose options refer to letters or
boxes drawn in the image. Inputs are one or more `items.jsonl` files with
`{"id", "source", "question", "options", "gold", "images"}` per line, where `gold` is an
option letter and image paths are absolute or relative to the file. `--records` adds
soft-target replay from public `TrainingRecord` files. About 5% of each items file (at most
`--dev-per-source`, chosen by ID hash) is held out for selection; benchmarks must never be
used for selection. Deltas are saved as `delta-E<epoch>.safetensors` twice per epoch.

`scripts/train_exit_head.py` records the raw residual after each `--layers` entry at the
answer and condition positions together with the full-depth option logits (optionally after
a `--delta`). It distills one `ExitHead` per layer with KL to the teacher plus
`--gold-weight` gold cross-entropy and writes `exit<L>.safetensors` with a ready
`early_exit` manifest block in `summary.json`. Mix-training deltas are a research format;
packaging them with an exit head as a released checkpoint is a separate, verified step.

## Avoid contamination

Keep train, dev, calibration and test assignments intact within a configuration. Do not merge
historical configurations: many contain the same observations. Old test groups were repeatedly
inspected and are regression material. Future research needs new untouched final populations.
Korean text in generated training examples is dataset content, not private project documents.
