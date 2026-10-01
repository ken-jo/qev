# Workflow recipes

## Select a tool or route

List the currently available handlers and what each does. Supply the user's actual goal
and relevant evidence. Ask a choice question; consume the chosen key only when the model
has not abstained and application policy permits the action. Missing evidence can route
to an explicit review candidate. A tool choice does not supply missing arguments or prove
that the tool completed its job.

## Image plus policy

Attach one photo and put the applicable rule in text, for example which visible materials
a facility accepts. A choice can identify the material, a score can assess visibility,
and a noul question can assess a specific acceptance condition. Ask separate questions
only when each answer is useful. A question cannot read another answer from the same call.

## Observe, judge, act, observe

Provide a recent screenshot or observed application state and legal candidate actions.
After selecting and executing an authorized action, capture the next state and verify the
result. Start with coarse visual conditions; tiny text and spatial games are documented
weaknesses. A model score alone is not evidence that a remote operation succeeded.

## Evaluate before relying on automation

Use held-out examples reflecting the real workflow. Record top-choice accuracy together
with abstention coverage, accepted-case accuracy and probability errors. Keep latency
measurements separate for model loading, warm inference, queueing and network. The
published photo timings are not a guarantee for screenshots or this skill's full loop.
Measure candidate-order changes and no-match cases; a schema-valid answer can be wrong.

The programming pattern was informed by the
[official TypeSafe skill](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md).
This is an independent QEV integration using QEV's own schema and local model; it does
not call Jev or claim equivalent capabilities.
