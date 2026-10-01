# Suggested release announcement

Introducing **Vision QEV**, an open research checkpoint for dynamic decisions over text and
images, built on **Qwen3.5-2B**.

Supply your own candidates and criteria; get `choice`, ordinal `score` or binary `noul`
probabilities and abstention in one batched forward, with zero generated answer tokens.
On a resident RTX 4060 Ti 8 GB, the measured serial photo workload achieved **114.94 ms
local HTTP p95**. A local playground includes sample photos and resolution comparisons.

The release includes code, adaptation weights, documented training data and evaluation
evidence. It is a research milestone: fresh procedural-workflow accuracy is 69.38%, while
OCR/spatial reasoning and uncertainty handling remain limited. The failed 2048 experiments
are documented openly.

Project: https://github.com/ken-jo/vision-qev

We are closing this Qwen-based line and using its lessons to investigate a separate
multimodal decision network. No JEV-equivalence claim is made.

## Short description

Qwen3.5-2B-based dynamic text/image decisions: request-defined candidates, choice/score/noul,
probabilities and abstention, with an open custom runtime and documented limitations.
