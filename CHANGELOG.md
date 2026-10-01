# Release history

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
