"""Command-line entry point; heavyweight imports are deferred to subcommands."""

import argparse
import json
from pathlib import Path

from veyra.constants import MODEL_ID, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(prog="veyra")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the pinned backbone and processor")
    download.add_argument("--cache-dir", default=".cache/huggingface")
    benchmark = commands.add_parser("profile", help="Measure resident-model request latency")
    benchmark.add_argument("--request", type=Path)
    benchmark.add_argument("--output", type=Path, default=Path("reports/profile.json"))
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--cache-dir", default=".cache/huggingface")
    benchmark.add_argument("--checkpoint")
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--iterations", type=int, default=20)
    infer = commands.add_parser("predict", help="Evaluate a request with a trained checkpoint")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--request", required=True, type=Path)
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--cache-dir", default=".cache/huggingface")
    serve = commands.add_parser("serve", help="Serve a trained checkpoint over a local HTTP API")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--image-root", required=True, type=Path)
    serve.add_argument("--device", default="cuda")
    serve.add_argument("--cache-dir", default=".cache/huggingface")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    build = commands.add_parser("build-data", help="Build the licensed starter training corpus")
    build.add_argument("--output", type=Path, default=Path("data/starter-v1"))
    build.add_argument("--scenes", type=int, default=512)
    build.add_argument("--text-cases", type=int, default=512)
    build.add_argument("--seed", type=int, default=20260930)
    build.add_argument("--no-beans", action="store_true")
    build.add_argument("--cache-dir", default=".cache/huggingface")
    augment = commands.add_parser("augment-data", help="Vary question count and phrasing by group")
    augment.add_argument("--records", type=Path, required=True)
    augment.add_argument("--output", type=Path, required=True)
    augment.add_argument("--seed", type=int, default=20261001)
    features = commands.add_parser("extract", help="Cache frozen Qwen features for head training")
    features.add_argument("--records", type=Path, required=True)
    features.add_argument("--output", type=Path, required=True)
    features.add_argument("--device", default="cuda")
    features.add_argument("--cache-dir", default=".cache/huggingface")
    train = commands.add_parser("train", help="Train the decision head from frozen features")
    train.add_argument("--records", type=Path, required=True)
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=0.0003)
    train.add_argument("--seed", type=int, default=19)
    calibration = commands.add_parser("calibrate", help="Fit held-out temperatures and abstention")
    calibration.add_argument("--records", type=Path, required=True)
    calibration.add_argument("--features", type=Path, required=True)
    calibration.add_argument("--checkpoint", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--device", default="cuda")
    calibration.add_argument("--max-error", type=float, default=0.15)
    evaluate = commands.add_parser("evaluate", help="Evaluate a trained head on a named split")
    evaluate.add_argument("--records", type=Path, required=True)
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=["dev", "calibration", "test"], default="dev")
    evaluate.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download

        print(
            snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=args.cache_dir,
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
            ),
            flush=True,
        )
    elif args.command == "profile":
        from veyra.benchmark import profile, write_fixture

        request = args.request or write_fixture(Path("data/profile"))
        profile(
            request,
            args.output,
            device=args.device,
            cache_dir=args.cache_dir,
            checkpoint=args.checkpoint,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    elif args.command == "augment-data":
        from veyra.augment_data import augment_dataset

        print(json.dumps(augment_dataset(args.records, args.output, args.seed), indent=2))
    elif args.command == "evaluate":
        from veyra.evaluate import evaluate_checkpoint

        evaluate_checkpoint(
            args.records, args.features, args.checkpoint, args.output, args.split, args.device
        )
    elif args.command == "calibrate":
        from veyra.calibrate import calibrate_checkpoint

        result = calibrate_checkpoint(
            args.records, args.features, args.checkpoint, args.output, args.device, args.max_error
        )
        print(json.dumps(result["calibration"], indent=2))
    elif args.command == "train":
        from veyra.training import train_head

        train_head(
            args.records,
            args.features,
            args.output,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.device,
            args.seed,
        )
    elif args.command == "extract":
        from veyra.features import extract_features

        print(
            json.dumps(
                extract_features(args.records, args.output, args.device, args.cache_dir), indent=2
            )
        )
    elif args.command == "build-data":
        from veyra.build_data import build_dataset

        print(
            json.dumps(
                build_dataset(
                    args.output,
                    args.scenes,
                    args.text_cases,
                    args.seed,
                    not args.no_beans,
                    args.cache_dir,
                ),
                indent=2,
            )
        )
    elif args.command in {"predict", "serve"}:
        from veyra.model import VeyraModel
        from veyra.schema import DecisionRequest

        model = VeyraModel.load(args.checkpoint, args.device, args.cache_dir)
        if args.command == "predict":
            request = DecisionRequest.from_json(args.request.read_text(encoding="utf-8"))
            print(json.dumps(model.predict(request), indent=2, ensure_ascii=False))
        else:
            import uvicorn

            from veyra.server import create_app

            uvicorn.run(create_app(model, args.image_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
