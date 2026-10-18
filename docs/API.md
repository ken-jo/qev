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

For a local checkpoint whose manifest declares serving `modes`,
`qev.QEV.load(checkpoint_dir, device="cuda", mode="fast")` selects a preset (per-image
pixel budget, optional early exit); `mode=None` uses the manifest `default_mode`. Such
predictions include a `mode` field. `early_exit=True` loads a declared early-exit head
directly. The released checkpoint declares neither, so `qev.load()` is unchanged. See
[architecture](ARCHITECTURE.md#serving-options-in-the-unreleased-source).

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

The public request schema is `qev.DecisionRequest`. The unreleased source extension
combines text with up to four ordered local images. Published SDK 0.2.1 accepts one image.
Image paths are resolved under the configured image root; remote URLs are not fetched
by the inference API. See [multi-image requests](MULTI_IMAGE.md).

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

Send JSON to `POST /v1/systemone`.

**Unreleased; source checkout only:** the endpoint also accepts a Jev/SystemOne-style
body: `state` as a plain string or structured JSON, `instructions` omitted (the question
ID is used) and `"model": "qev"`. The published SDK 0.2.1 requires the native format.
State objects containing `text` or `images` use strict native validation, including the
rejection of unknown fields. Other JSON states are serialized to text. Native requests
keep their validation rules, and images remain local paths.

Native `DecisionRequest.model` (including `qev predict --request`) remains
`"Qwen/Qwen3.5-2B"` or omitted; `"qev"` is normalized only by `/v1/systemone`.

`score.criteria` is an ordered list of descriptions;
the result is an expected zero-based level. `noul.criteria` has `true` and `false` strings.
The source maximum is four questions, 16 alternatives per choice/score and four images.
The configured token budget covers the complete request, including every image and every
question/view. Inputs exceeding it fail instead of silently truncating or dropping images.
`usage.input_images` reports the supplied image count in the source extension.

Inspect `answers`, `probabilities`, abstention fields and `usage` in the returned object.
Source predictions return `"model": "qev"`; `/v1/models` lists the same decision-model ID.
Both include a separate `backbone` object with the actual `model_id` and pinned `revision`.
The QEV skill accepts both `qev` and the SDK 0.2.1 response name `veyra-qwen3.5-2b`,
and accepts merged backbones with a nonempty ID and a 40-hex revision.
See `src/veyra/schema.py`, `src/veyra/probability.py` and `examples/request.json`.

The playground adds image-upload and resolution endpoints around the same model. A single
resident worker serializes inference. It is a trusted local demo; uploads are shared within
that server session. Local latency measurements do not imply concurrent serving capacity.
