# Local QEV playgrounds

Public Hugging Face Space creation was cancelled on 2026-10-01. No hosted Space was
created. The publication helper exits without making network or account changes.

The existing English Gradio application remains in `apps/hf_space/` for local use.
Its directory name records the earlier hosting experiment. It offers editable text/image
examples, six licensed sample photos, image resolution controls, choice/score/noul,
probabilities, abstention and request/response inspection.

Install Python 3.12, QEV and `gradio==6.29.0`; then run `python apps/hf_space/app.py`.
CPU is the default. For the GPU runtime, set `QEV_DEVICE=cuda`. The current checkpoint
and upstream revisions are pinned. A local checkpoint path can be supplied through
`QEV_CHECKPOINT_PATH`; the model cache through `QEV_CACHE_DIR`.

The original local interface in `apps/playground/` includes the historical 2048 experiment.
See [its instructions](../apps/playground/README.md).

## Timing and scope

CPU requests can take several seconds. GPU execution can reduce model processing time,
while input size, uploads, queues and startup affect the total wait. Runtime notices show
the server's configured device. The model itself uses BF16 backbone weights on GPU and
FP32 on CPU; readouts use FP32. Local CPU verification is not a GPU latency benchmark.

Sample photos overlap development data and are demonstrations, not evaluation results.
The public documentation remains English. No public hosted demo is required to install
or use the QEV Python package.
