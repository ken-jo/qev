# Prospective rule-execution targets

This is a data audit for a possible follow-up study. No model was trained, no inference was
performed, and the active continuation study and its candidate list remain unchanged.

`build_workflow_execution_targets.py` filters to the original procedural **train/development**
records before interpreting requests. It parses the rendered A–D definitions and priority rules,
enumerates all 16 assignments under their stated priors/sensor assumptions, and records:

- each primitive condition probability;
- each rule's probability of being true, before priority masking;
- the probability that each rule is the first match, including the default branch;
- the corresponding distribution over the caller's outcomes.

Summing first-match branches by their rendered outcomes reproduces all **5,040** eligible
questions' original candidate targets, with maximum absolute difference **2.22e-16**.
Equivalent typed views agree. No train/development group overlaps were found. The original
data bytes and parser source hashes are bound in `protocol.json`; `audit.json` binds the local
derived-target artifact. This parser is a training annotation tool, not a runtime decision solver.

| Population | Questions | Groups | Distinct states | Maximum rules / branches |
| --- | ---: | ---: | ---: | ---: |
| Training | 3,840 | 960 | 1,920 | 3 / 4 |
| Development | 1,200 | 300 | 600 | 2 / 3 |

## What the audit changes about the hypothesis

First-match labels are **not automatically richer supervision**. Training contains multiple
branches with the same outcome: 215 of 960 uncertain states retain extra branch entropy after
the outcome is known, averaging 0.0861 bits over all uncertain training states. Development's
branch outcomes are unique, so first-match labels there are exactly an outcome relabeling.
An improvement from another copy of that label would not establish better rule execution.

A useful follow-up would therefore distinguish primitive facts, individual rule satisfaction,
priority resolution and final decisions. Any model study needs a separately declared matched
control, a demonstration that auxiliary gradients reach the backbone, and the same independent
calibration/final gates. These computed targets alone establish neither learnability nor real
business generalization. Official teacher-labeled workflows do not have this exact execution
annotation; the audit covers the procedural subset only.

## Related work and the inference constraint

[SIM-CoT](https://arxiv.org/html/2509.20317v2) uses a training decoder to supervise intermediate
latent reasoning steps, removing that decoder for inference. Its method constructs latents
autoregressively and subsequently generates an answer. Our one-backbone-forward, zero-generated-
answer-token requirement would need a different implementation; this is not a reproduction or
a claim to its reported gains.

[Pause-token research](https://research.google/pubs/think-before-you-speak-training-language-models-with-pause-tokens/)
studies extra internal computation positions. The authors report that pause pretraining matters;
adding positions or only fine-tuning a standard model does not by itself establish an improvement.
Consequently, expanding Veyra's four positions remains a hypothesis, not a promised speed or
accuracy improvement.
