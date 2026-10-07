# Inference API

## Python SDK

The `qev` distribution contains both the SDK and the English playground. With Python
3.12, install the release wheel or the published PyPI distribution, then:

```python
import qev
from pathlib import Path

model = qev.load()  # Download missing model files; select CUDA when available.
request = qev.DecisionRequest.from_json(open("request.json", encoding="utf-8").read())
result = model.predict(request, image_root=Path("."))
print(result["answers"])
```

The first load retrieves about 4.6 GB of pinned model files. Later loads reuse the cache.
`qev.load(device="cpu", cache_dir="/path/to/hub", offline=True)` selects CPU and requires
complete local files. `QEV_HOME` sets the checkpoint/cache root; `QEV_CACHE_DIR` overrides
the Hub cache. The lower-level `qev.QEV.load(...)` remains available for explicit checkpoint
management and retains the original inference contract.

## Commands

```sh
qev playground
qev download
qev predict --request request.json
qev serve --image-root .
```

All inference commands prepare missing model files automatically. Use `--offline` to
require a complete cache. `predict` resolves image paths relative to the request JSON;
`serve` resolves them under `--image-root`. The API server defaults to
`http://127.0.0.1:8000`; the playground defaults to `http://127.0.0.1:7860`.
See [playground installation, uv and GPU instructions](PLAYGROUND.md).

## Request and response

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

Send JSON to `POST /v1/systemone`. The endpoint also accepts a Jev/SystemOne-style body:
`state` as a plain string or any JSON value (JSON is serialized to text), `instructions`
omitted (the question ID is used) and `"model": "qev"`. Bodies valid under the native schema
are handled exactly as before; the response is unchanged and images are still local paths. `score.criteria` is an ordered list of descriptions;
the result is an expected zero-based level. `noul.criteria` has `true` and `false` strings.
The maximum is four questions, 16 alternatives per choice/score and one image. Processed
inputs exceeding the configured token budget fail instead of silently truncating.

Inspect `answers`, `probabilities`, abstention fields and `usage` in the returned object.
The model ID remains `Qwen/Qwen3.5-2B` in the validated contract to identify the actual base.
See `src/veyra/schema.py`, `src/veyra/probability.py` and `examples/request.json`.

The playground adds image-upload and resolution endpoints around the same model. A single
resident worker serializes inference. It is a trusted local demo; uploads are shared within
that server session. Local latency measurements do not imply concurrent serving capacity.
