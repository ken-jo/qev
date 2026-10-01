# QEV agent skill

The English `qev` skill teaches an agent to define a narrow judgment, call the local QEV
model, read its typed result, and apply application policy before choosing a next step.
It supports text, one image, or both. See [SKILL.md](../skills/qev/SKILL.md).

## Install

Install the Python runtime with Python 3.12:

```sh
python -m pip install qev==0.2.1
```

Copy the complete `skills/qev` folder from this repository, or the `qev` folder inside
`qev-skill-0.2.1.zip`, into your agent's skills directory. For Codex this installation uses
`$CODEX_HOME/skills/qev`, or `~/.codex/skills/qev` when `CODEX_HOME` is unset. Restart or
reload the agent's skill catalog after installation. The skill includes references,
scripts, and photo assets; copying only SKILL.md omits its helper.

The pip package installs the SDK/playground runtime. The agent skill is a separate folder;
installing or importing the SDK does not alter an agent's configuration automatically.

## Use

Examples to give the agent:

> Use $qev to route this support request among billing, technical support and review.

> Use $qev to judge this photo against my sorting policy. Return the likely material,
> a visibility score, and the probability that the acceptance condition is met.

> Use $qev to choose among these available next actions. Observe the result before
> requesting another decision, and keep ambiguous cases for review.

For repeated calls, start a resident server in another terminal:

```sh
qev serve --image-root <evidence-directory>
```

Then the agent uses the helper at its actual installed skill path:

```sh
python <skill-dir>/scripts/decide.py --request <request.json> --dry-run
python <skill-dir>/scripts/decide.py --request <request.json>
```

Use `--endpoint http://127.0.0.1:8765/v1/systemone` with the earlier QEV playground.
That interface requires photos to be uploaded through its UI/API first. Standard
`qev serve` resolves image paths under `--image-root`. The helper also supports one-off
`--backend sdk` calls; these load the model in each process.

Exit 0 means a model decision, 2 means review, and 1 means an error. Probabilities remain
in the raw response. Model abstention is preserved; an optional application threshold can
add review. A model decision does not authorize an external action.

The implementation was informed by the
[official TypeSafe agent skill](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md),
using QEV's own schema and runtime. It does not call Jev or increase the model's underlying
accuracy. See [verification scope](../release/skill-verification.json).
