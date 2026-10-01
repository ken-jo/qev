# Inference API

The public request schema is `qev.DecisionRequest`. Text and one local image can be
combined. Image paths are resolved under the configured image root; remote URLs are not
fetched by the inference API.

```json
{
  "state": {"text": "The customer was charged twice and asks for a refund."},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle the request?",
      "criteria": {"billing": "Payments and refunds", "technical": "Software faults"}
    }
  }
}
```

Send JSON to `POST /v1/systemone`. `score.criteria` is an ordered list of descriptions;
the result is an expected zero-based level. `noul.criteria` has `true` and `false` strings.
The maximum is four questions, 16 alternatives per choice/score and one image. Processed
inputs exceeding the configured token budget fail instead of silently truncating.

Inspect `answers`, `probabilities`, abstention fields and `usage` in the returned object.
The model ID remains `Qwen/Qwen3.5-2B` in the validated contract to identify the actual base.
See `src/veyra/schema.py`, `src/veyra/probability.py` and `examples/request.json`.

The playground adds image-upload and resolution endpoints around the same model. A single
resident worker serializes inference. It is a trusted local demo; uploads are shared within
that server session. Local latency measurements do not imply concurrent serving capacity.
