"""Public inference CLI; training commands remain in the compatibility runtime."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="qwen3.5-classification")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the pinned Qwen3.5-2B backbone")
    download.add_argument("--cache-dir", default=".cache/huggingface")
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
        from huggingface_hub import snapshot_download

        from veyra.constants import MODEL_ID, MODEL_REVISION

        print(
            snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=args.cache_dir,
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
            )
        )
        return
    import torch

    from qwen3_5_classification import DecisionRequest, QwenClassification

    torch.set_num_threads(4)
    if args.command == "predict":
        model = QwenClassification.load(
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

        model = QwenClassification.load(
            args.checkpoint,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=True,
            merge=True,
        )
        app = create_app(model, args.image_root)
        uvicorn.run(app, host=args.host, port=args.port)
