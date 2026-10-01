# Packaged English playground

The English UI, SDK, image presets and license notices are included in the `qev`
Python distribution. The source lives in `src/qev/playground/`.

```sh
qev playground
# Or, from this source checkout:
uv run qev playground
```

QEV prepares the pinned model on first use and reuses the cache afterwards. CUDA is
selected when available; use `--device cpu` or `--device cuda` to choose explicitly.
The default address is http://127.0.0.1:7860. For a trusted network, add `--host 0.0.0.0`.

`app.py` is a compatibility launcher for earlier local commands. No public Space is
created. See [the playground guide](../../docs/PLAYGROUND.md) for installation, caches,
offline use, request limits, and included sample-image provenance.

[GitHub: ken-jo/qev](https://github.com/ken-jo/qev)
