"""Public inference CLI; training commands remain in the compatibility runtime."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="qev")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download QEV checkpoint and pinned backbone")
    download.add_argument("--cache-dir", default=".cache/huggingface")
    download.add_argument("--output", type=Path, default=Path("checkpoints/qev"))
    download.add_argument("--checkpoint-only", action="store_true")
    download.add_argument("--base-only", action="store_true")
    for name in ("predict", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--cache-dir", default=".cache/huggingface")
        if name == "predict":
            command.add_argument("--request", type=Path, required=True)
        else:
            command.add_argument("--image-root", type=Path, required=True)
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "download":
        from qev.download import download_backbone, download_checkpoint

        if args.base_only and args.checkpoint_only:
            parser.error("--base-only and --checkpoint-only cannot be combined")
        result = {}
        if not args.base_only:
            result["checkpoint"] = str(download_checkpoint(args.output, args.cache_dir))
        if not args.checkpoint_only:
            result["backbone_cache"] = download_backbone(args.cache_dir)
        print(json.dumps(result, indent=2))
        return
    import torch

    from qev import QEV, DecisionRequest

    torch.set_num_threads(4)
    if args.command == "predict":
        model = QEV.load(
            args.checkpoint,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=True,
            merge=True,
        )
        request = DecisionRequest.from_json(args.request.read_text(encoding="utf-8"))
        print(json.dumps(model.predict(request, args.request.parent.resolve()), indent=2))
    else:
        import uvicorn

        from veyra.server import create_app

        model = QEV.load(
            args.checkpoint,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=True,
            merge=True,
        )
        app = create_app(model, args.image_root)
        uvicorn.run(app, host=args.host, port=args.port)
