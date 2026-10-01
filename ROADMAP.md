# Research roadmap

## Closed Qwen-based line: Qwen3.5 Classification 0.1.0

Freeze the evaluated Qwen3.5-2B adaptation, publish the custom runtime, licensed dataset
snapshots, exact evidence and known failures. Future bug fixes must preserve or explicitly
revalidate the frozen inference contract. Do not market this checkpoint as a new base network.

## Separate next network

After this publication is complete, begin a distinct experiment with an original integrated
decision network. Its name and results will be separate from this checkpoint.

1. Establish independent perception tests: local object identity, digits, positions, relations
   and conflicting evidence. Reserve unseen renderers, objects, compositions and photo domains.
2. Prototype image tokens retaining spatial coordinates, text/criterion encoders, candidate-
   conditioned evidence attention and one shared candidate scorer. Include typed ordinal and
   binary outputs; test candidate permutation and paraphrase invariance.
3. Compare against the frozen Qwen3.5 Classification baseline at equal data and hardware budgets. Use Qwen
   representations or teachers only where a measured benefit justifies them; backbone choice
   is open. Separate perception, evidence binding, decision logic and probability errors.
4. Study supervised proper scoring, distillation and RL only after perception works. Select
   on development data, fit calibration separately and evaluate once on an untouched final set.
5. Target resident latency below 300 ms on RTX 4060 Ti, with a declared image/token/question
   workload. Report p50/p95, warm/cold conditions and throughput separately.

This is a research plan, not a claim that the proposed network already exists or meets these
goals. LAYA's interface and candidate scoring are references; proprietary JEV internals are
unknown. Success requires measurable held-out transfer, not an impressive single game demo.
