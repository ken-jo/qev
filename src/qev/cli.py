"""SDK, local playground and first-use model downloads in one installed command."""

import argparse
import json
import os
import sys
from pathlib import Path

from qev import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qev")
    parser.add_argument("--version", action="version", version=f"qev {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the pinned model for later use")
    download.add_argument("--cache-dir")
    download.add_argument("--output", "--checkpoint", type=Path)
    download.add_argument("--checkpoint-only", action="store_true")
    download.add_argument("--base-only", action="store_true")
    download.add_argument(
        "--offline", action="store_true", default=os.environ.get("QEV_OFFLINE") == "1"
    )
    for name in ("predict", "serve", "playground"):
        command = commands.add_parser(name)
        command.add_argument(
            "--checkpoint", type=Path, help="Checkpoint folder; default: QEV cache"
        )
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        command.add_argument("--cache-dir", help="Hugging Face cache; also accepts QEV_CACHE_DIR")
        command.add_argument(
            "--offline", action="store_true", default=os.environ.get("QEV_OFFLINE") == "1"
        )
        if name == "predict":
            command.add_argument("--request", type=Path, required=True)
        else:
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=7860 if name == "playground" else 8000)
            if name == "serve":
                command.add_argument("--image-root", type=Path, default=Path.cwd())
            else:
                command.add_argument(
                    "--open", action="store_true", help="Open the playground in a browser"
                )
    for name in ("star", "support"):
        command = commands.add_parser(name, help=f"Show the QEV {name} page")
        command.add_argument("--open", action="store_true", help="Open the page in your browser")
    args = parser.parse_args(argv)
    if args.command == "download" and args.base_only and args.checkpoint_only:
        parser.error("--base-only and --checkpoint-only cannot be combined")
    if hasattr(args, "port") and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"qev: {error}\n")


def run(args):
    if args.command in {"star", "support"}:
        from qev.community import REPOSITORY_URL, SUPPORT_URL

        url = REPOSITORY_URL if args.command == "star" else SUPPORT_URL
        print(url)
        if args.open:
            import webbrowser

            webbrowser.open(url)
        return
    from qev.runtime import cache_home, load_model, prepare_checkpoint, resolve_cache

    def progress(message):
        print(message, file=sys.stderr, flush=True)

    if args.command == "download":
        from qev.download import CHECKPOINT_REVISION, download_backbone, download_checkpoint

        folder = args.output or cache_home() / "checkpoints" / CHECKPOINT_REVISION
        cache = resolve_cache(args.cache_dir)
        if args.checkpoint_only:
            result = {
                "checkpoint": str(download_checkpoint(folder, cache, local_files_only=args.offline))
            }
        elif args.base_only:
            result = {"backbone_cache": download_backbone(cache, local_files_only=args.offline)}
        else:
            folder, cache = prepare_checkpoint(
                folder, cache_dir=cache, offline=args.offline, progress=progress
            )
            result = {"checkpoint": str(folder), "backbone_cache": cache}
        print(json.dumps(result, indent=2))
        return
    if args.command == "playground":
        from qev.playground.app import launch

        launch(
            checkpoint=args.checkpoint,
            device=args.device,
            cache_dir=args.cache_dir,
            offline=args.offline,
            host=args.host,
            port=args.port,
            inbrowser=args.open,
        )
        return
    import torch

    from qev import DecisionRequest

    torch.set_num_threads(4)
    request = None
    if args.command == "predict":
        request = DecisionRequest.from_json(args.request.read_text(encoding="utf-8"))
    elif not args.image_root.is_dir():
        raise ValueError("--image-root must be an existing directory")
    model = load_model(
        args.checkpoint,
        device=args.device,
        cache_dir=args.cache_dir,
        offline=args.offline,
        progress=progress,
    )
    if request is not None:
        print(json.dumps(model.predict(request, args.request.parent.resolve()), indent=2))
    else:
        import uvicorn

        from veyra.server import create_app

        uvicorn.run(create_app(model, args.image_root), host=args.host, port=args.port)
