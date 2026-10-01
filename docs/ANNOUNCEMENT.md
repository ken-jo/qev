# Suggested release announcement

Introducing **QEV**, an open multimodal decision model inspired by **LAYA** and built
with **Qwen3.5-2B's text and vision backbone**.

Supply your own candidates and criteria; get `choice`, ordinal `score` or binary `noul`
probabilities and abstention in one batched forward, with zero generated answer tokens.
On a resident RTX 4060 Ti 8 GB, the measured serial photo workload achieved **114.94 ms
local HTTP p95**. A local playground includes sample photos and resolution comparisons.

LAYA inspired the typed decision interface; Qwen provides the multimodal backbone.
The release includes code, adaptation weights, documented training data and evaluation
evidence. It is a research milestone: fresh procedural-workflow accuracy is 69.38%, while
OCR/spatial reasoning and uncertainty handling remain limited. The failed 2048 experiments
are documented openly.

In a matched 2,000-question text evaluation, QEV scored 77.00% and LAYA's specialist
76.95%, with no demonstrated accuracy advantage. On two tasks absent from QEV's audited
adaptation data, QEV reached 82.50% on news topics and 50.25% on emotion, below the
specialist's 95.25% and 60.00%. LAYA was smaller and faster. The full comparison includes
probability errors, training exposure, confidence intervals and negative results.

Project: https://github.com/ken-jo/qev

We are closing this Qwen-based line and using its lessons to investigate a separate
multimodal decision network. No JEV-equivalence claim is made.

## Short description

LAYA-inspired decisions with Qwen3.5-2B vision: request-defined candidates, choice/score/noul,
probabilities and abstention, with open weights, a custom runtime and documented evaluations.
