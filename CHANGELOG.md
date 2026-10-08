# Release history

## Unreleased

- `POST /v1/systemone` accepts Jev/SystemOne-style request bodies (string or JSON `state`,
  optional `instructions`, `"model": "qev"`); native requests and responses are unchanged.
- State objects containing `text` or `images` retain strict native validation. Mixed
  objects with additional fields are rejected rather than converted into photo-free text.
- The compatibility extension is available from source; published SDK 0.2.1 remains native-only.
- Playground controls remain English even when the browser language is Korean or
  another supported locale; multilingual model inputs remain available.
- Model card: highlights, request format and file tables, `pip install qev`, citation, and
  earlier Veyra names moved to a Provenance section. The published Hugging Face README is
  not yet updated.

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
