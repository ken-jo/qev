# Runtime setup

Use Python 3.12 and the released QEV SDK. If it is already installed, reuse that environment.

```sh
python -m pip install https://huggingface.co/ken-jo/qev/resolve/main/runtime/qev-0.2.1-py3-none-any.whl
qev serve --image-root <evidence-directory> --host 127.0.0.1 --port 8000
```

Keep the server in a separate terminal/process. First use downloads missing QEV and pinned
Qwen files (about 4.6 GB); later launches reuse them. `GET /health` reports readiness.
`--device auto` selects CUDA if available; otherwise CPU. A missing CUDA build requires
installing matching torch/torchvision CUDA wheels as described in the
[playground guide](https://github.com/ken-jo/qev/blob/main/docs/PLAYGROUND.md).
Run one inference at a time on this research server. Keep its loopback default unless the
user requests network access. Remote callers must use a user-selected server; the helper
has no automatic cloud fallback or authentication mechanism.

For an existing Python environment, invoke the skill helper with that environment's
Python. For an isolated uv environment, `--with` installs the same SDK for the helper:

```sh
uv run --python 3.12 --with https://huggingface.co/ken-jo/qev/resolve/main/runtime/qev-0.2.1-py3-none-any.whl python <skill-dir>/scripts/decide.py --request <request.json>
```

For one call without a server:

```sh
python <skill-dir>/scripts/decide.py --backend sdk --request <request.json> --device auto
```

SDK mode loads QEV once for that process; use HTTP or retain `qev.load()` in a Python
application for repeated judgments. `QEV_HOME` and `QEV_CACHE_DIR` select caches. SDK mode
supports `--offline`, `--checkpoint`, `--cache-dir` and `--image-root`. HTTP image paths
resolve under the running server's image root, not the helper's working directory.

Use `--endpoint <full-systemone-URL>` for a chosen server. `--timeout` applies to HTTP
waiting. `--min-confidence` may add a review floor; it cannot remove the model's abstention.
`--review-label review` marks a choice of that explicit candidate as requiring review.
Neither flag replaces domain validation. No default extra threshold is imposed.
