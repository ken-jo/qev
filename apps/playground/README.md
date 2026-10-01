# Vision QEV playground

The local playground uses the released Qwen3.5-2B-based checkpoint and keeps it resident
on the GPU. It includes a Korean interface, uploaded or preset photos, choice/score/noul
examples, image-resolution comparisons, actual API JSON and an exploratory 2048 page.
The examples are demonstrations, not additional evaluation data.

From the repository root, after the main README's installation and checkpoint download:

```sh
uv run python apps/playground/server.py --checkpoint checkpoints/vision-qev --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. Use `--host 0.0.0.0` for a trusted network and open
`http://<model-PC-IP>:8765`; inbound firewall policy still applies. The page appears while
the GPU model loads. The server requires the pinned Qwen weights to be cached already.

Windows helpers are available:

```powershell
powershell -ExecutionPolicy Bypass -File apps/playground/start.ps1 -NoBrowser
powershell -ExecutionPolicy Bypass -File apps/playground/stop.ps1
```

The default checkpoint path is `checkpoints/vision-qev`. Tailscale helpers can configure a
private HTTPS proxy with an exact peer allowlist. They must run under the user's own account;
no saved device identity or peer configuration is distributed.

## Inputs and measurements

- Upload JPG, PNG or WebP (up to 20 MB), or choose one of six licensed TrashNet samples.
- Choose a long-edge resolution. Each variant is derived from the original, preserving ratio.
  Source labels and filenames are not inserted into model input as answers.
- A score preset returns an expected level, not image accuracy. Report both the typed value
  and its probability distribution.
- Model inference time and browser round-trip are separate. Resolution comparisons run
  serially. They are local diagnostics, not fresh benchmark estimates.

## 2048

The image demo provides rendered board pixels and the declared instructions/history, without
the numeric matrix or search results. Detailed instructions can be passed to the step API.
The default stops on abstention; exploratory forced execution is recorded explicitly.
The model has no game-specific training. Its published 72-game study achieved zero wins.
Read [the study](../../reports/text2048/assistance-study-v1/README.md).

The local server serializes GPU inference. Uploaded images are shared session resources,
not isolated by user. It has no public multi-tenant authentication. Use a service gateway
with authentication and isolation before offering an internet-facing product.
