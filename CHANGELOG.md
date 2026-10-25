# Release history

## Unreleased

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
  earlier Veyra names moved to a Provenance section. The Hugging Face README was updated
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
- Preserved prior publication evidence under `release/history/0.1.0/`.

## 0.1.0 — Qwen3.5 Classification research release

- Renamed the public project from Veyra to Qwen3.5 Classification and explicitly attributed Qwen3.5-2B.
- Preserved the selected v13 adaptation weights, calibrated manifest and original runtime.
- Added the `qwen3_5_classification` Python facade and `qwen3.5-classification` inference command.
- Curated English release documentation, source-specific dataset licensing and clean public commits.
- Preserved numerical success/failure evidence, including the unsuccessful 2048/image-cell study.
- Archived personal Korean documents and original development history privately.

This is the final snapshot of the current Qwen-based research line. Version 0.1.0 denotes
the renamed distribution, not a reset of training or an assertion of production readiness.
