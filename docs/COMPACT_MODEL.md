# Compact QEV research direction

## The size requirement

The released QEV checkpoint retains a 2.213B-parameter inference backbone. Its 7.992M
stored adaptation/readout parameters are not its full model size. The next architecture
should target **300-450M total inference parameters**, including vision, text, fusion
and decision readouts. This is a proposed engineering budget, not an achieved model.

The [LAYA model card](https://huggingface.co/convaiinnovations/laya) reports 421M parameters
for its ModernBERT-large English model and 322M for its multilingual model. Both are
text models. Their size follows from their chosen encoder architecture. LAYA defines
RLCD as Reinforcement Learning for Calibrated Decisions, optimizing probability reports
with proper scoring rules. That training objective does not remove layers or parameters.

## Architecture proposal

Use a compact pretrained text encoder and a compact vision encoder, connected through
one trainable fusion and decision network. Candidate descriptions remain part of the
request. The head scores candidate-conditioned evidence rather than a fixed label list.
The model supports text-only requests and image-plus-text requests in the same network.

Allocate the parameter budget across both encoders and fusion before selecting weights.
Avoid retaining a 1.88B language decoder simply because its output generation is unused.
A caption-then-text pipeline would add an error-prone intermediary representation and
does not meet the goal of direct integrated visual decisions.

## Training sequence

1. Establish supervised text, image and image-plus-policy performance in the compact
   architecture using licensed data and human or deterministic ground truth.
2. Distill useful probability distributions and representations from larger teachers,
   retaining ground-truth supervision. QEV's documented visual and uncertainty failures
   must not become unquestioned teacher targets.
3. Compare RLCD-style probability training against cross-entropy, Brier and ordinal
   proper-scoring objectives under matched data and compute budgets. RLCD is a proposed
   experiment, not a claimed feature of the current released checkpoint.
4. Calibrate and select abstention on a separate split. Quantize only after a reliable
   smaller architecture is established.

Quantization reduces bytes per parameter, not the number of parameters. Structured
pruning can reduce dimensions, but requires recovery training and accuracy checks.
Distillation trains a genuinely smaller student network; its quality is not guaranteed.

## Release gates

- Count all parameters, model files and peak serving memory; report merged and unmerged
  sizes separately where adapters exist.
- Preserve request-defined candidates and choice/score/noul behavior.
- Use disjoint source, image, group and task-template splits, with untouched final tests.
- Report accuracy, distribution error, NLL, ECE, ordinal MAE and selective risk/coverage.
- Measure text and image latency on the same RTX 4060 Ti workload and include cold start
  and concurrency separately. Retain the earlier 300 ms target as a measurement goal.
- Test candidate permutations, unseen policies and visual failures rather than relying
  on narrow training-family improvements.

Implementation and training of this compact architecture are separate from the QEV
0.1.1 naming and Python-package release. No compact weights have been produced yet.
