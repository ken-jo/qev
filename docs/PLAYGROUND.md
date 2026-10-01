# English public playground

The English Spaces application is in [`apps/hf_space/`](../apps/hf_space/README.md).
It provides seven editable text/image examples, six licensed sample photos, image-resolution
selection, `choice` / `score` / `noul`, candidate probabilities, abstention and request/response
inspection. Samples overlap development data and are explicitly labeled as demonstrations.

The frozen model, inference modules and calibration are unchanged. The public demo adapter
limits each request to one image and one question. The model API still supports four questions.
CPU mode uses the original float32 runtime path; CUDA uses the original bfloat16 path.

## Response time

The notice identifies the server's execution mode: CPU, GPU, or shared GPU. Inference
runs on the server; visitors do not need a GPU in their own computer. CPU requests can
take several seconds. GPU hosting can reduce model processing time, but the full wait
also includes upload, network, queue, and any startup or shared-GPU allocation delay.
Input length, image resolution, and candidate count affect the work required.

The result panel reports model processing time only. No speedup ratio or Hugging Face
response-time guarantee is established by the local GPU measurements. A shared GPU can
finish computation faster while still taking longer overall when its queue is busy.

## Readiness checks

The source test suite includes request validation, all preset schemas, image resizing and
rejection of unsupported candidates. Actual local CPU inference is exercised with a text
request and all three photo decision types. The Gradio HTTP queue is also checked with real
requests, including photo upload. These checks do not establish deployment availability or
performance on Hugging Face hardware.

## Hosting

As of 2026-10-01, Hugging Face's documentation says CPU Basic has no hourly hardware charge,
but creating a Gradio/Docker Space requires a paid plan. Eligible free personal accounts
can host up to two ZeroGPU Spaces; email verification and an account older than 30 days
are required. The account API remains the authority on actual eligibility.

Visitors can use an existing ZeroGPU Space without a PRO subscription. Daily GPU-time
allowances and queue priority depend on the visitor's account tier; these are separate
from the requirements for creating a Space. A paid dedicated GPU is a separate hosting
choice, and this project's publisher does not select or purchase one.

See [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) and
[ZeroGPU eligibility](https://huggingface.co/docs/hub/spaces-zerogpu).

After the publishing account is eligible, the maintainer can run:

```sh
python scripts/verify_playground.py --url http://127.0.0.1:7860/
python scripts/publish_playground.py --hardware cpu-basic
# Or, for an eligible free ZeroGPU account:
python scripts/publish_playground.py --hardware zero-a10g
```

The publisher accepts only these hardware tiers, checks the original photo hashes and
English source, and requires successful local HTTP inference evidence. It does not change
subscriptions or request paid GPU hardware. After upload, inspect the Space's build/runtime
status and exercise its public API before advertising it as live.

No public-PC tunnel or private-network connection is needed. Hosting is independent of
the maintainer's resident GPU playground. Temporary uploads expire; the app does not add
visitor images or text to the released dataset.
