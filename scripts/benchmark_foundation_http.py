"""Measure real TCP requests with distinct held-out photographs and resident weights."""

import argparse
import hashlib
import json
import socket
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import torch
import uvicorn
from PIL import Image
from train_foundation_head import write
from validate_model import latency_summary

from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.model import VeyraModel
from veyra.server import create_app


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists() or args.iterations < 20:
        raise ValueError("use a new output and at least 20 measured photographs")
    pool = [
        record
        for record in read_records(args.records)
        if record.split == "test"
        and "domain:image_waste" in record.tags
        and "view:recognition" in record.tags
    ]
    groups = {}
    for record in sorted(pool, key=lambda r: hashlib.sha256(r.id.encode()).hexdigest()):
        groups.setdefault(record.group_id, record)
    chosen = list(groups.values())[: args.iterations + 3]
    if len(chosen) != args.iterations + 3:
        raise ValueError("insufficient distinct photograph groups")
    if any(
        len(r.request.state.images) != 1
        or len(r.request.questions) != 1
        or len(candidates_for(next(iter(r.request.questions.values())))) != 6
        for r in chosen
    ):
        raise ValueError("benchmark requires one photo, one question and six candidates")
    root = args.records.parent.resolve()
    dimensions = []
    for record in chosen:
        with Image.open(root / record.request.state.images[0].path) as image:
            dimensions.append(list(image.size))
    protocol = {
        "weights_sha256": digest(args.checkpoint / "head.safetensors"),
        "manifest_sha256": digest(args.checkpoint / "manifest.json"),
        "dataset_sha256": digest(args.records),
        "script_sha256": digest(Path(__file__)),
        "record_ids": [r.id for r in chosen],
        "image_dimensions": dimensions,
        "scope": (
            "One real photograph, one recognition question, six candidates; "
            "local HTTP round trip including JSON and image decode"
        ),
        "warmups": 3,
        "photos_per_request": 1,
        "questions_per_request": 1,
        "candidates_per_question": 6,
        "resident": True,
        "feature_cache": False,
        "serial": True,
        "excludes": ["model loading", "WAN transport", "concurrent load"],
        "selection": "Fixed ID hash ordering, unique image groups; no correctness filtering",
    }
    write(args.output.with_suffix(".protocol.json"), protocol)
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = VeyraModel.load(args.checkpoint, local_files_only=True)
    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(model, root), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    timings, tokens = [], []
    try:
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("local server did not become ready")
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False
        ) as client:
            for index, record in enumerate(chosen):
                torch.cuda.synchronize()
                start = time.perf_counter()
                response = client.post("/v1/systemone", json=record.request.model_dump())
                response.raise_for_status()
                result = response.json()
                elapsed = 1000 * (time.perf_counter() - start)
                if result["usage"]["output_tokens"] != 0:
                    raise ValueError("unexpected generated tokens")
                if index >= 3:
                    timings.append(elapsed)
                    tokens.append(result["usage"]["input_tokens"])
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        sock.close()
    result = {
        "protocol": protocol,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "load_seconds": load_seconds,
        "requests": len(timings),
        "latency_ms": latency_summary(timings),
        "p95_target_met": bool(np.percentile(timings, 95) <= 300),
        "input_token_range": [min(tokens), max(tokens)],
        "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    write(args.output, result)
    print(
        json.dumps(
            {
                "requests": len(timings),
                "p50": result["latency_ms"]["p50"],
                "p95": result["latency_ms"]["p95"],
                "target_met": result["p95_target_met"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
