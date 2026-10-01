# Architecture and compatibility

Vision QEV 0.1.0 freezes the Qwen-based research line. Base identity and revision are fixed
in `src/veyra/constants.py`. We use Qwen's vision encoder and language model, merged learned
language LoRA, a pretrained-letter-initialized 16-position readout and a learned conditioned
binding residual. Four internal positions support condition supervision.

The encoder consumes each question together with its evidence and candidate descriptions.
Questions form one inference batch. No output sentence or chain of thought is decoded.
The selected options are mapped back to the caller's keys, calibrated by question type,
and processed by the learned abstention thresholds. Score is an expectation over ordered
levels. Noul is the probability of true under the supplied proposition criteria.

## Public rename without model changes

The distribution and command are `vision-qev`; `vision_qev` exposes `VisionQEV` and
`DecisionRequest`. The 38 original `veyra` runtime modules remain byte-identical to the
evaluated 0.3.0a4 wheel. `VeyraResult:` is a trained input marker and must not be renamed.
The legacy `veyra` CLI, checkpoint field names and historical evidence identifiers remain
for compatibility. These internal names do not denote a separate running model.

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
