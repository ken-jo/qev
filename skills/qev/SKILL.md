---
name: qev
description: Use QEV for bounded text and image decisions, including routing among supplied actions, evaluating visual conditions, and scoring ordered levels with probabilities and abstention. Use when building or running a QEV decision workflow.
---

# QEV decisions

Turn current evidence and explicit candidates into a typed judgment using the local
QEV SDK or a running QEV server. The host agent defines the task, supplies observations,
interprets uncertainty, and executes any authorized next step. QEV produces no generated
explanation. Its Qwen3.5-2B backbone and learned decision heads are documented in the
[model card](https://huggingface.co/ken-jo/qev).

## Choose the judgment

| Need | Type | Consume |
| --- | --- | --- |
| Select one supplied handler, label, or next action | `choice` | `choice` and the candidate distribution |
| Assess one proposition about evidence | `noul` | `noul`, the probability of true |
| Assess a degree on explicit ordered levels | `score` | `score`, the expected zero-based level |

Use ordinary code for known rules, arithmetic, and checking which actions are available.
Use QEV where understanding the evidence requires a semantic judgment. For a workflow
with repeated decisions, keep a resident server so each call avoids model loading.

## Prepare and call

1. Read [the request contract and examples](references/contract.md) when composing a new
   request. Put observed facts and relevant policy in `state.text`; attach at most one
   image. Keep evidence separate from the instructions that define the judgment.
2. Supply a complete candidate set with distinct descriptions. Include a review/no-match
   option if the task needs one. For action selection, offer only actions available in
   the observed state. Do not infer hidden state or insert a desired answer into evidence.
3. Keep requests within four questions, 2-16 candidates per choice/score and the model's
   processed-token budget. Group independent questions about the same evidence; dependent
   questions need a later request with the preceding result.
4. Use the bundled helper to validate JSON and call QEV. Resolve its location relative to
   this skill folder, not the user's working directory:

```sh
python <skill-dir>/scripts/decide.py --request <request.json> --dry-run
python <skill-dir>/scripts/decide.py --request <request.json>
```

The default endpoint is `http://127.0.0.1:8000/v1/systemone`. For setup or a first
invocation, read [runtime setup](references/runtime.md). A single SDK call is also
available with `--backend sdk`; it loads the model for that process.

## Interpret before acting

- Preserve the returned distribution, `abstained`, and calibration fields. If QEV abstains,
  report review status and collect useful evidence or use the host agent's reasoning.
  Label host reasoning separately; do not invent replacement QEV probabilities.
- `noul=0.2` means a 20% estimate of true, not low confidence in a chosen label. A score is
  an expected level, not a confidence percentage. Use the full score distribution when
  decisions depend on tails or extremes.
- The helper keeps model abstention and optionally adds a task-specific confidence floor
  or explicit review labels. Such thresholds require evaluation on the user's domain.
  A result marked `decided` records the model judgment; it does not grant action permission.
- After an action changes the scene or application, observe again before the next judgment.
  An execution failure, invalid response, or unavailable model produces an error, not a
  simulated decision. The helper never switches models or retries automatically.

For integration patterns and evaluation boundaries, read
[workflow recipes](references/workflows.md). English has the strongest evaluation coverage.
OCR, fine spatial reasoning and visual 2048 remain weak; this skill does not change weights
or establish Jev-equivalent accuracy or latency.
