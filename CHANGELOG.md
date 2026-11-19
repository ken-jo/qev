# Release history

## Unreleased

- Opt-in fast image path, all off by default (outputs unchanged): requests may carry
  in-memory HWC RGB `uint8` numpy arrays or torch tensors as `{"pixels": ...}` images (Python
  only; not part of the JSON schema); `OptionModel.load(..., device_preprocess=True)` resizes
  and normalises images on the GPU with the stock processor and grid; `vision_graph=True`
  replays a per-grid CUDA graph of the vision tower (bitwise equal to eager; small LRU); and
  `image_max_pixels`/`image_min_pixels` override the per-image budget after the serving
  mode, including on bundled `accurate`-only checkpoints. Serving modes may also set
  `image_min_pixels`. Without CUDA the flags fall back to the stock path.

## 0.2.2 — SDK runtime as qev_runtime

- SDK runtime now ships as qev_runtime; legacy veyra modules removed; no model weight change.
  The model package keeps `head.safetensors` and `manifest.json` byte-identical to 0.2.1 and
  adds `runtime/qev-0.2.2-py3-none-any.whl`; packaged text and photograph inference is
  recorded in `release/model-inference.json`. The earlier wheels and the evidence and
  release files that name the legacy runtime leave the model package; they
  remain at Hugging Face revision `1b801c0a`. The agent skill installs the 0.2.2 wheel.
- The research runtime moves into this repository as the companion package `qev_runtime`
  (a sibling of `qev` and `qev_decision_index`); its separate console command is retired.
  The public `qev` package keeps only the SDK surface: CLI, download, playground and
  serve. The package version is the QEV version (`qev.__version__`). Code, scripts, docs and apps use QEV names; corpus
  generators, the playground headers (`X-QEV-Playground`, `X-QEV-Inference-Ms`) and the
  status app id (`qev-playground`) now write QEV names. The trained result marker keeps its
  original bytes in `qev_runtime/frozen_identifiers.py`; the packed-prompt markers are now
  `[QEV_CANDIDATE]` and `[QEV_QUESTION]` (the published option-readout checkpoints do not
  use them).
- Removed the pre-QEV research history: the Foundation v11, Workflow v12/v13 and
  release-comparison evidence, `release/history`, the 2026-10-08 handoff record, the 15
  frozen release scripts and the other scripts and configurations that only produced or
  audited that evidence, the corpus record-ID salts, and the pre-rename compatibility names (the SDK 0.2.1
  response model name and the old playground header in the skill and the legacy playground,
  and the old bundle loader name). The `archive/pre-qev` tag keeps them.
- Removed the 2048 game: the playground engine and its `/2048` and `/api/2048/*` routes, the
  text/visual 2048 run, comparison, audit and readout scripts, their test, and the
  `reports/text2048` and `reports/visual2048` studies. The docs no longer cite 2048 results.
- Packaging: the research CLI is installed as `qev-research` (it already used that program
  name), and `qev`, `qev_runtime` and `qev_decision_index` ship PEP 561 `py.typed` markers.
- `qev_decision_index:QEVEngine`: Decision Index vision engine for the released QEV-0.8B and
  QEV-2B checkpoints (one forward pass, native resolution up to 1.6 MP per image, at most 4
  images and 8,192 input tokens; longer requests are reported as unsupported). Needs
  `decision-index[vision]` installed.
- Source predictions and `/v1/models` use the decision-model name `qev`, with backbone
  provenance reported separately. The QEV skill also accepts the legacy SDK 0.2.1 response
  name and validates pinned merged-backbone metadata.
- `train_option_mix.py` evaluation keeps failed questions in each source's denominator and
  reports `completed`/`failed` counts. Its `select` scores are not directly comparable with
  earlier run records that excluded failed requests; compare only after scoring both sets
  with the same denominator policy.

- Option checkpoint manifests may request per-image budgets up to 1,638,400 pixels
  (1.6 MP); the default remains 262,144.
- Opt-in `fast_jpeg_decode` encoder setting (default off, so published outputs are
  unchanged): large JPEGs are decoded at a reduced libjpeg DCT scale that still covers the
  processor's per-image budget before the usual resize; the 40 MP guard and audit
  fingerprints are kept.
- Optional early exit: a manifest `early_exit` block and `OptionModel.load(...,
  early_exit=True)` serve a truncated text stack with a distilled `ExitHead`. Off by
  default; manifests without the field load exactly as before.
- Optional CUDA graphs: `OptionModel.load(..., cuda_graph=True)` replays captured graphs of the
  text stack for single requests up to 768 tokens (right-padded to a length bucket); longer
  requests run eagerly. Off by default. Short text requests ran 1.6-2.4x faster on an RTX 4060 Ti;
  probabilities can differ from eager by bfloat16 rounding.
- Option head widths follow the backbone's text `hidden_size`, so hidden-1024
  (Qwen3.5-0.8B) and hidden-2048 backbones load; the released 2B head layout is unchanged.
- Option manifests may name a pinned repository of merged full weights
  (`"weights": "merged"`, 40-hex revision) as the backbone and may omit `adaptation` when
  no LoRA is stored with the heads. Existing manifests load as before.
- Optional manifest serving `modes` (per-image pixel budget and early exit) with
  `default_mode`, selected by `OptionModel.load(..., mode=...)`; predictions report the
  active `mode` only for checkpoints that declare modes.
- Research recipes `scripts/train_option_mix.py` (resolution-elastic, option-permuted mix
  training) and `scripts/train_exit_head.py` (exit-head distillation). Results are not yet
  published.

- Source SDK, HTTP request schema and playground accept up to four ordered images.
  Both encoder paths include every image, with explicit image numbers for multi-image
  requests; zero/one-image prompt formatting is preserved.
- File-root checks, decode/file limits and cleanup apply to every input image; the
  configured request token budget remains shared across all images and questions/views.
- Replay and refresh overlap audits inspect all images, including later list entries.
- Multi-image input is a source feature with unmeasured accuracy, calibration and latency.
  Published SDK 0.2.1 and current training executions are unchanged.

- `POST /v1/systemone` accepts Jev/SystemOne-style request bodies (string or JSON `state`,
  optional `instructions`, `"model": "qev"`); native request validation is unchanged.
- State objects containing `text` or `images` retain strict native validation. Mixed
  objects with additional fields are rejected rather than converted into photo-free text.
- The compatibility extension is available from source; published SDK 0.2.1 remains native-only.
- Playground controls remain English even when the browser language is Korean or
  another supported locale; multilingual model inputs remain available.
- Model card: highlights, request format and file tables, `pip install qev`, citation, and
  earlier project names moved to a Provenance section. The Hugging Face README was updated
  on 2026-10-08 with a separate documentation-only publication receipt; model and SDK
  artifacts remain unchanged.

## 0.2.1 — English playground and multilingual labeling

- Unified the Windows launcher with the packaged English playground.
- Retired the 2048 web interface while preserving its recorded research results.
- Replaced the model card's English/Korean language tags with Multilingual and documented
  the English-focused evaluation scope.
- Preserved the QEV 0.1.1 weights, calibration, inference modules and dataset records.

## 0.2.0 — SDK and local playground

- Bundled the English playground, six licensed photos and all preset files in the wheel.
- Added `qev playground` and `qev.load()` with pinned model preparation on first use.
- Added persistent caches, offline operation and automatic CUDA/CPU selection.
- Kept `qev predict`, `qev serve`, `qev download` and the low-level `QEV.load` API.
- Added voluntary GitHub Star and support links; browser opening requires `--open`.
- Preserved the QEV 0.1.1 learned model, calibration and all 38 inference modules.

## 0.1.1 — QEV packaging release

- Renamed the public GitHub project, Hugging Face model and dataset to QEV / qev-data.
- Added the `qev` Python package, `QEV` interface and checkpoint-aware download command.
- Prepared a PyPI wheel and source distribution with a GitHub Trusted Publisher workflow.
- Documented measured tensor dtypes, stored adaptation size and exact backbone counts.
- Cancelled public Space creation; retained the existing local playground source.
- Preserved the trained weights, calibration, 38 inference modules and dataset records.
- Preserved prior publication evidence under `release/history/0.1.0/` (now archived at the
  `archive/pre-qev` tag).

## 0.1.0 — Qwen3.5 Classification research release

- Renamed the public project from Veyra to Qwen3.5 Classification and explicitly attributed Qwen3.5-2B.
- Preserved the selected v13 adaptation weights, calibrated manifest and original runtime.
- Added the `qwen3_5_classification` Python facade and `qwen3.5-classification` inference command.
- Curated English release documentation, source-specific dataset licensing and clean public commits.
- Preserved numerical success/failure evidence, including the unsuccessful 2048/image-cell study.
- Archived personal Korean documents and original development history privately.

This is the final snapshot of the current Qwen-based research line. Version 0.1.0 denotes
the renamed distribution, not a reset of training or an assertion of production readiness.
