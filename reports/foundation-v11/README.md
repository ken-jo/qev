# Veyra Foundation evidence

This directory contains aggregate reports and frozen protocols. Original photographs and text
datasets are downloaded separately. Local `runs/`, `data/` and checkpoint paths in the reports
describe the exercised workspace; the scripts and source snapshots preserve reproduction inputs.

## Data and training

- `source-downloads.json` and `data-protocol.json`: source revisions, licenses, corpus hashes,
  grouped splits, duplicate handling and exact train/development/calibration/test counts.
- `head-protocol.json`, `head-comparison.json`, `head-selection.json`: five training objectives,
  three seeds each, development selection and the selected readout checkpoint.
- `head-training-fit.json`: selected readout training accuracy; not held-out performance.
- `backbone-protocol.json`, `backbone-history.json`: subsequent common-backbone LoRA adaptation.
- `midpoint-amendment.json`: the single parameter midpoint added after half-epoch development
  feedback and before calibration or final inference. It was not part of the initial protocol.
- `merged-dev-0.json` through `merged-dev-3.json` and `selection.json`: all deployment-path
  candidates, their development guardrails and the chosen candidate.
- [Source snapshots](source-snapshots/README.md): exact historical bytes of generators that
  embedded their own source hash before later formatting-only changes.

## Final evaluations

- `baseline-final.json` / `foundation-final.json`: identical 3,866-question corpus before and
  after adaptation. Fresh and legacy groups, domains and question views are reported separately.
- `paired-comparison.json`: paired group bootstrap differences against Dynamic v2 and the two
  LAYA checkpoints on the shared text subset. These descriptive intervals are not adjusted for
  multiple comparisons. Veyra was adapted on these task families; LAYA was not.
- `typed-regression.json`: official typed-decisions test, 2,000 teacher-labeled questions,
  previously inspected and excluded from this training/selection/calibration.
- `typed-baseline-standardized.json` and `typed-regression-audit.json`: preserved Dynamic v2
  predictions rescored with the current metric definitions, paired against the new checkpoint.
  All gold labels and target distributions were checked. The observed accuracy decline is
  49.05% to 46.85%, with a paired 95% interval of −3.65 to −0.75 percentage points.
  Reproducing this historical audit requires the original local prediction archive; a fresh
  baseline can instead be run with `evaluate_foundation.py` on the same typed-regression corpus.
- `legacy-policy-regression.json`: the old 3,576-question synthetic final set, reused only for
  regression evaluation. It is no longer a fresh final benchmark.
- `calibration.json`: temperature and abstention fitting on disjoint calibration groups.
- `runtime.json`: resident runtime matrix, dynamic criteria probes, actual loopback HTTP,
  zero generated tokens and request-contract checks.
- `photo-http.json`: 40 distinct held-out TrashNet photographs, six candidates, one question,
  serial loopback HTTP, three warmups; model loading and WAN/concurrency excluded.

## LAYA comparison details

Use `laya-official-original-base.json`, `laya-official-original-specialist.json` and
`laya-official-original-base-full-input.json` for the final official benchmark comparison.
Original question dictionaries preserve the reference SDK's defaults when criteria are absent.

The earlier `laya-official-base.json` and `laya-official-specialist.json` are retained as diagnostics.
Those runs inserted Veyra's default truth descriptions into 200 originally criteria-free questions.
They must not be substituted for the corrected original-input results.

The stock base context limit truncates state text in 34 official questions. The 1,024-token
diagnostic changes only the context limit; it preserves released weights and temperatures.
The corresponding input audits document zero truncation at 1,024 tokens, zero truncation for the
specialist and zero truncation on the foundation text corpus. LAYA receives no image input.

`laya-foundation-base.json` and `laya-foundation-specialist.json` use the same 2,141 text questions
as Veyra Foundation. This includes soft uncertainty targets, so the hard-label accuracy denominator
is smaller. The specialist's official workflow training differs from Veyra's foundation adaptation.

`export-manifest.json` binds the exported evaluation reports to their source files and SHA-256.
Release packaging and post-publication verification reports were generated afterward and are
listed separately below.

`package-verification.json` records real text/image inference from the bundled wheel under
Python isolated mode, one backbone forward per request, and zero generated tokens. It reuses the
installed dependencies and cached Qwen base. `release-package.json` records the verified archive
hash, the preserved synthetic-policy and latency targets, and the explicitly unmet workflow
specialist parity. Hugging Face publication is false; the folder is prepared for upload.

`public-download-verification.json` records an anonymous download of the published GitHub
release archive, its matching SHA-256, and successful text/photo inference using the wheel
extracted from that download. The release tag and asset remain fixed; this later report does
not change the published model. Existing dependencies and the cached Qwen base were reused.

`hf-documentation-update.json` records the later model-card and roadmap revision in the local
Hugging Face preparation. Its 40-file package adds English and Korean roadmaps; weights,
calibration, runtime wheel, loader and evaluation reports match the original release byte for
byte. The checksum manifest is new. The earlier inference and download checks above still refer
to the original 38-file archive; the documentation revision did not rerun model inference.
