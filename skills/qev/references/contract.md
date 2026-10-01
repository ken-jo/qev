# QEV request and response

QEV SDK 0.2.0 exposes the frozen model 0.1.1 contract. Validate with
`qev.DecisionRequest.from_json(...)`; do not copy another provider's wire format.

```json
{
  "state": {"text": "The customer was charged twice and requests a refund."},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments, invoices and refunds",
        "technical": "Software faults",
        "review": "The evidence does not establish a suitable team"
      }
    }
  }
}
```

`state` accepts `text` and an optional `images: [{"path": "item.jpg"}]`. Image paths refer
to local files under the configured image root. HTTP calls send a path, not image bytes;
the file must exist on the server. Remote image URLs are unsupported. For local SDK calls,
use `--image-root` or place the JSON beside its referenced image.

For a score, `criteria` is a list of descriptions from level 0 upward. For noul it is an
object with `false` and `true` descriptions. Every question has a string `instructions`.
The optional `model` field, if present, must be `Qwen/Qwen3.5-2B` to identify the backbone.
QEV does not accept arbitrary nested state fields or generated-output schema definitions.

The included examples are `assets/text-request.json` and
`assets/photograph/request.json`. The photograph example asks choice, score and noul
questions about one image in one forward pass. Its MIT photo overlaps prior development
and verification; it is a functional example, not an independent accuracy benchmark.

Response answers are keyed by the input question IDs:

```json
{
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.75, "technical": 0.10, "review": 0.15},
      "confidence": 0.75,
      "confidence_definition": "maximum_candidate_probability",
      "abstained": true,
      "calibrated": true,
      "abstention_policy_fitted": true
    }
  }
}
```

The response above is a schema illustration, not a measured QEV prediction. Noul answers
have `noul` and `false`/`true` probabilities. Score answers have `score`, numeric-string
probability keys and a `legend`. `confidence` is the maximum candidate probability for all
three types; for noul, it can be the probability of false. `usage.output_tokens` is zero.

The helper outputs `model_called`, overall `status`, per-question `decisions`, and the
unmodified QEV `response`. Exit 0 means decided, 2 means at least one question needs review,
and 1 means validation, transport or inference error. A dry run has `model_called: false`
and no prediction. Its validation checks schema, not model token length or image contents.
