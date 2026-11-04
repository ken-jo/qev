# Real embedding-adapter and typed-decision baseline

This implementation is `qev.pemli_real`; the earlier `qev.pemli_jev` is a
synthetic verifier and its outputs are not evidence of trained-model performance.

## Scope

- Load actual EmbeddingGemma 2 with a pinned 40-character HF revision and no audio tower.
- Decode local image bytes using QEV's bounded image loader; never embed image filenames.
- Use contextual token outputs. Upstream HF already projects `last_hidden_state`
  to `text_config.embedding_dim` (768); do not add another 512-to-768 projection.
- Freeze the upstream model for an initial 8 GB GPU training baseline. Train a shared
  token projection (`embedding.safetensors`) and candidate-specific evidence verifier
  (`decision.safetensors`) with soft-target cross entropy on actual training records.
  The embedding artifact is a task adapter plus a pinned backbone reference, not a
  newly pretrained standalone foundation model. No LoRA, distillation, MRL quality,
  conformal guarantees, quantization, or performance advantage is claimed.
- Candidate descriptions, question tokens, and image/text evidence enter learned
  cross-attention. Candidate identities and enumeration positions never enter the model.
- Use the native `DecisionRequest`, `candidates_for`, `Calibration`, and `typed_answer`
  contracts, including score's list rubric and noul's false/true order.
- Public predict and POST `/v1/systemone` return typed decision JSON, probabilities,
  calibration and abstention flags, with zero generated answer tokens. Internal vectors
  are not exposed. Uncalibrated checkpoints conservatively mark all answers abstained;
  the selected choice / expected score / true probability remain in the JSON.

## Input and execution

Use an existing QEV `TrainingRecord` JSONL with train and untouched test splits,
relative local image paths, source provenance, targets, and group IDs. The loader checks
split leakage and image checksums. Do not concatenate overlapping historical corpora.
`vision-mix-2026-10` source manifests are not directly TrainingRecord JSONL; rebuild
and convert that source explicitly before using this pipeline.

On the ken Windows host, from this repository:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pemli_windows.ps1 `
  -Records C:\path\to\records.jsonl `
  -BaselineCheckpoint C:\path\to\qev08-checkpoint `
  -BackboneRevision <40-character-HF-commit> `
  -OutputRoot runs/pemli-real-001
```

The script uses an isolated Python 3.12 environment, checks CUDA, trains and saves,
fits an explicitly empirical abstention policy on a separate calibration split,
then reloads the calibrated checkpoint in a new process for test evaluation. QEV-0.8B
loads in another process to avoid holding both backbones in GPU memory. Baseline
weights must already be downloaded; its manifest must identify a 0.8B backbone.
Dependencies differ from the older repository lock because EmbeddingGemma 2 needs
newer Transformers. The script records installed versions. Windows CUDA and baseline
compatibility still require validation on the target host.

Model inference:

```powershell
python -m qev.pemli_real predict --checkpoint runs/pemli-real-001/checkpoint --request examples/request.json --image-root examples --device cuda
python -m qev.pemli_real serve --checkpoint runs/pemli-real-001/checkpoint --image-root examples --device cuda --port 8001
```

## Evidence and limitations

The calibration split is divided by group for temperature fitting and empirical
threshold selection. This is not a conformal risk guarantee. Missing/insufficient
question types remain always-abstained. Test groups are checked against both training
and calibration provenance.

`training.jsonl` records actual optimizer-step losses. The manifest stores data checksum,
training groups, backbone revision, and objective. Embedding and decision adapters are
saved separately with safetensors and reloaded for evaluation. Evaluation outputs are
immutable, include per-question answers and paired subset hashes, and report accuracy,
NLL, Brier, ECE, score MAE, coverage, accepted error, resident p50/p95 request latency,
and peak allocated/reserved CUDA bytes. Comparison uses paired group bootstrap intervals.
Model loading and one warmup are excluded; latency is serial with PEMLI state cache off.
The script does not establish equal image preprocessing budgets with QEV-0.8B; record
that difference before interpreting latency. Original QEV-0.8B training overlap with the
chosen test population must be audited separately. No real checkpoint or GPU results
are supplied with this source change.

Offline tests exercise gradient propagation, optimizer changes, checkpoint roundtrip,
permutation equivariance, evidence/question sensitivity, pixel cache invalidation, and
three-type JSON/HTTP contracts. Test encoders are injected only in tests; they do not
verify pretrained visual quality. Full image sensitivity and task accuracy require the
actual backbone and target GPU run.
