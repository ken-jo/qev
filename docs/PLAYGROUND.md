# QEV playground and SDK

The `qev` Python wheel includes the English playground, its presets, six sample photos,
license notices and the SDK. You do not need to clone the repository to use the UI.

## Install

Python 3.12 is required. The public runtime wheel is available with the model:

```sh
python -m pip install https://huggingface.co/ken-jo/qev/resolve/main/runtime/qev-0.2.1-py3-none-any.whl
qev playground
```

Or use uv's isolated tool environment:

```sh
uvx --python 3.12 --from https://huggingface.co/ken-jo/qev/resolve/main/runtime/qev-0.2.1-py3-none-any.whl qev playground
```

From a source checkout:

```sh
uv sync --frozen
uv run qev playground
```

Open http://127.0.0.1:7860. `--open` opens that page automatically. Use `--port 7870`
for another port. `--host 0.0.0.0` makes the interface reachable through the machine's IP
on a trusted network. No public hosted Space is created.

## First launch and caches

The first model use fetches the checksum-verified QEV adaptation and pinned Qwen3.5-2B
backbone: about 4.6 GB of model files, in addition to Python dependencies. A progress
message appears while the model is being prepared. Later launches use the same cache;
starting from a different working directory does not trigger another model download.

| Setting | Purpose |
| --- | --- |
| `QEV_HOME` | QEV checkpoint/cache root; Windows defaults to `%LOCALAPPDATA%/qev`, Linux/macOS to the XDG cache or `~/.cache/qev` |
| `QEV_CACHE_DIR` or `--cache-dir` | Override the Hugging Face model cache |
| `--checkpoint` | Use an explicit folder containing the released QEV head and manifest |
| `--offline` or `QEV_OFFLINE=1` | Require complete local files; never fetch missing files |
| `qev download` | Prepare both model components without opening the UI |

`qev predict`, `qev serve` and Python's `qev.load()` use the same bootstrap path. The
low-level `QEV.load()` API is retained for callers that manage checkpoint loading directly.
An existing checkpoint with a different checksum is rejected and preserved.

## CPU and GPU

The default `--device auto` uses CUDA when it is available, otherwise CPU. Use
`--device cuda` to require CUDA or `--device cpu` to select CPU. The source checkout's
uv lock selects CUDA 12.8 wheels. For uvx, add `--torch-backend cu128`
before `--from` to use that CUDA wheel index. For pip GPU installations:

```sh
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

CPU inference uses FP32 and can take seconds. CUDA uses a BF16 backbone with FP32
readouts. UI timings report model computation; uploads, startup and queueing add to the
user's wait. Published RTX 4060 Ti timings have their own recorded workload and scope.

## Included interface

- Editable text, image and image-plus-policy examples.
- Six unmodified TrashNet sample photographs with MIT notices and hash provenance.
- `choice`, ordered `score`, and `noul` decisions.
- Image long-edge controls from 64 to 1,024 pixels, preserving aspect ratio.
- Probabilities, abstention and exact request/response JSON.
- Optional Star and support links. Installing or running QEV never stars a repository.

Each UI request contains one question and at most one image; the SDK supports up to
four questions. The model allows 2-16 candidates and a 2,048 processed-token budget.
Requests run one at a time, with up to eight waiting. Uploads are limited to 10 MB / 16
megapixels. Per-inference temporary images are removed; upload caches expire after
approximately ten minutes and are swept every five minutes. Analytics is disabled by default.

Sample photographs overlap development data and are examples, not independent benchmarks.
The packaged implementation and asset notices live in `src/qev/playground/`.

## Windows launcher and earlier experiments

`apps/hf_space/app.py` and `apps/playground/start.ps1` launch the same packaged English
interface. The Windows helper uses port 8765 by default for existing local bookmarks;
`qev playground` uses port 7860 by default. Both accept an explicit port.

The earlier Korean server is retained for experimental API reproducibility. It is not
the supported playground. The 2048 game has been removed from the playground; its
recorded evaluation results remain in the research reports. See the
[Windows launcher instructions](../apps/playground/README.md).

[GitHub: ken-jo/qev](https://github.com/ken-jo/qev)
