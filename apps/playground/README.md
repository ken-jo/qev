# QEV playground launcher

The supported playground is the English interface shipped in the `qev` package.
It provides text and image decisions, `choice` / `score` / `noul`, sample photos,
image resolution controls, probabilities, abstention and request/response JSON.
The 2048 game has been retired from the playground.

## Run

```sh
qev playground
```

This opens a server at http://127.0.0.1:7860. For the earlier local address:

```sh
qev playground --host 0.0.0.0 --port 8765
```

On Windows, from a checkout with `uv sync --frozen` completed:

```powershell
powershell -ExecutionPolicy Bypass -File apps/playground/start.ps1 -NoBrowser
powershell -ExecutionPolicy Bypass -File apps/playground/stop.ps1
```

The helper uses port 8765 and loopback by default. Add `-BindAddress 0.0.0.0` for
access through the model PC's IP. Optional arguments are `-Port`, `-Device`,
`-Checkpoint`, `-CacheDir`, and `-Offline`. The helper launches the packaged English
application from this checkout's source and records its process for the stop helper.
The first use downloads the pinned model if it is not already cached.

See [the package playground guide](../../docs/PLAYGROUND.md) for installation,
GPU requirements, cache settings and input limits. The interface processes one
question per request; the SDK supports up to four questions.

## Historical research code

The earlier `server.py`, its Korean static interface and the 2048 experiment code
are retained only for reproducing historical API experiments. The Windows start
helper no longer launches that server, its home page no longer links to 2048, and
its `/2048` page returns HTTP 410. The supported English application has no game
route or game controls. Recorded evaluations remain available in the
[research report](../../reports/text2048/assistance-study-v1/README.md).
