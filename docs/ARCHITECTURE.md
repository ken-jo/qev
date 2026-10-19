# Architecture and compatibility

QEV 0.1.1 freezes the Qwen-based research line. Base identity and revision are fixed
in `src/veyra/constants.py`. We use Qwen's vision encoder and language model, merged learned
language LoRA, a pretrained-letter-initialized 16-position readout and a learned conditioned
binding residual. Four internal positions support condition supervision.

The encoder consumes each question together with its evidence and candidate descriptions.
Questions form one inference batch. No output sentence or chain of thought is decoded.
The selected options are mapped back to the caller's keys, calibrated by question type,
and processed by the learned abstention thresholds. Score is an expectation over ordered
levels. Noul is the probability of true under the supplied proposition criteria.

## Public rename without model changes

The distribution and command are `qev`; `qev` exposes `QEV` and
`DecisionRequest`. Published SDK 0.2.1 retains the evaluated inference modules.
`VeyraResult:` is a trained input marker and must not be renamed.
The legacy `veyra` CLI, checkpoint field names and historical evidence identifiers remain
for compatibility. These internal names do not denote a separate running model.

## Ordered multi-image source extension

The unreleased source supports zero to four images in both packed and option encoder
paths. Multiple images receive `Image 1:`, `Image 2:`, etc. boundaries before their
individual visual placeholders. Every question/view receives the complete image group
in the same order. Zero/one-image prompt construction retains its previous format.
The learned weights, base revision and configured token/pixel budgets are unchanged.
Replay and refresh audits check every image for overlap with excluded observations.
Multi-image runtime correctness, accuracy, calibration and latency need independent
validation; historical single-image results do not qualify this extension.

## Serving options in the unreleased source

These options are off or unchanged by default; the released manifest produces the same
decoded pixels and outputs as before.

- **Resolution.** An option checkpoint manifest may set `encoder.image_max_pixels` up to
  1,638,400 (1.6 MP). The default per-image budget remains 262,144 pixels, and a larger
  budget also spends more of the shared request token budget.
- **JPEG draft decode (opt-in).** With `"fast_jpeg_decode": true` in the manifest
  `encoder` block (`OptionConfig` or, for the packed path, `EncoderConfig`; default
  `false`), a JPEG with more than twice the processor's per-image budget is decoded by
  libjpeg at a 1/2, 1/4 or 1/8 DCT scale chosen so that the decoded image still
  covers the budget. The processor resize then sets the final image grid as before. The
  40 megapixel limit is checked on the original size first, other formats are unaffected,
  and audit fingerprints still hash the full-resolution decode. Decoded pixels are close
  to, but not bit-identical with, a full decode followed by the processor resize.
- **Backbone-sized heads and merged backbones.** The readout, condition readout,
  binding head and exit head take their width from the backbone text configuration's
  `hidden_size` (1024 for Qwen3.5-0.8B, 2048 for Qwen3.5-2B); the released 2B head layout
  is unchanged. The manifest `backbone` block is either the pinned Qwen3.5-2B or
  `{"model_id": "<repo>", "revision": "<40-hex commit>", "weights": "merged"}` for a
  repository whose root holds full merged Qwen3.5 weights with config, tokenizer and
  processor files. The `adaptation` block may be absent or empty when no LoRA is stored
  with the heads, for example because it is already merged into those weights. The `qev`
  SDK download command still fetches only the pinned released checkpoint and Qwen3.5-2B;
  other checkpoints load through `OptionModel.load`.
- **Early exit.** A manifest may declare
  `"early_exit": {"layer": 16, "head": "exit16.safetensors", "sha256": "..."}`
  (`sha256` optional). `OptionModel.load(..., early_exit=True)` keeps only the first
  `layer` text layers, removes the final text norm and replaces the readout, condition
  readout and binding head with an `ExitHead`: LayerNorm(2*hidden), Linear(2*hidden, 1024),
  GELU and Linear(1024, 16) over the raw residual at the answer position concatenated with
  the `VeyraResult:` condition position. `hidden` comes from the backbone configuration.
  Early exit requires internal condition slots, is inference-only (the truncated model
  cannot be saved) and reuses the checkpoint calibration, which was fitted on full-depth
  logits. Exit heads are produced by `scripts/train_exit_head.py`; see
  [training](TRAINING.md#mix-training-and-early-exit-recipes). Accuracy and latency of
  these options require separate measurement.
- **Serving modes.** One checkpoint can offer named presets, for example
  `"modes": {"accurate": {"image_max_pixels": 1638400}, "fast": {"image_max_pixels":
  262144, "early_exit": true}}` with an optional `"default_mode": "accurate"`. A mode may
  only set `image_max_pixels` (between the encoder's `image_min_pixels` and 1,638,400) and
  `early_exit` (which requires the manifest `early_exit` block).
  `OptionModel.load(..., mode="fast")` selects a preset; `mode=None` uses `default_mode`
  when present and otherwise the manifest encoder settings. An explicit `early_exit=`
  argument must agree with the selected mode. Predictions from a checkpoint with modes
  carry a `mode` field (`null` when none is selected); checkpoints without modes,
  including the released one, load and answer exactly as before. Models loaded in a mode
  are for inference only.

The public source history starts with curated release commits. Original author copyright,
source hashes, experiment identifiers and negative results remain attributable. Prior
development history and personal Korean notes are archived privately.

## Relationship to other systems

[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) supplies the base network and weights.
[LAYA](https://huggingface.co/convaiinnovations/laya) inspired request-defined candidates
and typed decisions. JEV's public interface motivated fast probabilistic decisions.
Neither LAYA nor JEV code or weights are included. No affiliation, original-base-network
claim or equivalent-performance claim is made.

An original multimodal network is a separate future research line described in
[the roadmap](../ROADMAP.md). It is not part of this released checkpoint.
