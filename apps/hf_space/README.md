---
title: Qwen3.5 Classification
emoji: "Q"
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
models:
- ken-jo/qwen3.5-classification
datasets:
- ken-jo/qwen3.5-classification-data
short_description: Dynamic text and image decisions with Qwen3.5-2B
---

# Qwen3.5 Classification playground

English interface for the released Qwen3.5-2B adaptation. Upload a photograph, choose a
licensed example or supply text, then define a choice, ordinal score or binary noul task.
The candidates and criteria are defined by the request, rather than fixed semantic labels.

The model and base revisions are pinned. The 38 evaluated inference modules, adaptation
weights and calibration remain unchanged. CPU execution uses the runtime's original
float32 path; the published GPU measurements used bfloat16. Small numerical differences
between devices are possible. This demo is not a new accuracy or latency benchmark.

## Running

Install `requirements.txt` under Python 3.12 and run `python app.py`. CPU is the default.
The first launch downloads the public model and the pinned Qwen base. For an existing
cache, set `QEV_CACHE_DIR`. Maintainers can set `QEV_CHECKPOINT_PATH` to a verified local
checkpoint and `QEV_OFFLINE=1` for local verification. `QEV_DEVICE=cuda` selects a local GPU.
For eligible ZeroGPU Spaces, select ZeroGPU hardware and set `QEV_ZERO_GPU=1`.

The notice reflects the server's CPU, GPU, or shared GPU mode. GPU hosting can reduce
model processing time; end-to-end response time also depends on input size, uploads,
startup and queues. Visitors do not need a GPU in their own computer. Shared ZeroGPU
hosting has daily usage limits and can involve a wait for available GPU capacity.

Public requests are serialized, with up to eight waiting requests. One image and one question
are accepted per request, with at most 16 candidates and the runtime's 2,048-token budget.
Image uploads are limited to 10 MB / 16 megapixels. Per-inference temporary files are deleted;
Gradio upload caches are swept every five minutes after a ten-minute expiry. Hosting logs
and network handling remain subject to Hugging Face policies.

Samples are six unmodified TrashNet images, with their original MIT notice and hash
provenance in `samples/`. They overlap development sources and are not independent tests.
Source category labels are not inserted into inference requests as answers.

## Links

- [Model and evaluation](https://huggingface.co/ken-jo/qwen3.5-classification)
- [Training data](https://huggingface.co/datasets/ken-jo/qwen3.5-classification-data)
- [Source code](https://github.com/ken-jo/qwen3.5-classification)
- [Maintainer GitHub](https://github.com/ken-jo)
- [Maintainer LinkedIn](https://www.linkedin.com/in/ik-chan-jo)

The public Space has no connection to the maintainer's PC or private network. No paid
hardware is required by this package; actual free hosting eligibility is determined by
Hugging Face. Queue and network latency are distinct from local model computation time.
