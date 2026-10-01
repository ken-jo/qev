"""Run resident Qwen inference, semantic probes, timing matrix and a real HTTP server."""

from __future__ import annotations

import argparse
import copy
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
from PIL import Image, ImageDraw

from veyra.data import read_records
from veyra.model import VeyraModel
from veyra.schema import DecisionRequest
from veyra.server import create_app


def latency_summary(values):
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("validation reports are immutable; use a new output path")
    if args.iterations < 20:
        raise ValueError("use at least 20 measured fresh-image requests per condition")
    root = args.records.parent.resolve()
    fixtures = root / "validation-fixtures"
    fixtures.mkdir(exist_ok=True)
    colors = [
        "blue",
        "red",
        "green",
        "yellow",
        "purple",
        "orange",
        "black",
        "pink",
        "brown",
        "cyan",
        "gray",
        "magenta",
        "beige",
        "maroon",
        "navy",
        "teal",
    ]
    for size in (384, 1024):
        for i in range(args.iterations + 3):
            with Image.new("RGB", (size, size * 2 // 3), "white") as image:
                draw = ImageDraw.Draw(image)
                x, y = 15 + i % 13, size // 4 + i % 11
                draw.rectangle((x, y, x + size // 5, y + size // 5), fill=colors[i % 4])
                image.save(fixtures / f"{size}-{i}.png")
    start = time.perf_counter()
    model = VeyraModel.load(args.checkpoint, local_files_only=True)
    load_seconds = time.perf_counter() - start
    manifest = json.loads((args.checkpoint / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "checkpoint_weights_sha256": manifest["weights_sha256"],
        "checkpoint_manifest_sha256": hashlib.sha256(
            (args.checkpoint / "manifest.json").read_bytes()
        ).hexdigest(),
        "load_seconds": load_seconds,
        "gpu": torch.cuda.get_device_name(),
        "scope": "JSON parse, image decode, processor, Qwen, trained head, JSON serialize",
        "feature_cache": False,
        "serial_requests": True,
        "warmups_per_condition": 3,
        "excludes": ["model loading", "HTTP transport (reported separately)"],
        "latency_target_ms": 300,
        "matrix": {},
        "protocol": {},
        "probes": [],
    }

    def run(raw):
        return model.predict(DecisionRequest.model_validate(raw), image_root=root)

    def request(i=0, size=384, count=4, multi=False):
        choices = colors[:count] if count != 2 else [colors[i % 4], colors[(i + 1) % 4]]
        raw = {
            "state": {
                "text": "Use the picture as evidence.",
                "images": [{"path": f"validation-fixtures/{size}-{i}.png"}],
            },
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "What is the color of the leftmost object?",
                    "criteria": {f"c{j}": color for j, color in enumerate(choices)},
                }
            },
        }
        if multi:
            raw["questions"].update(
                {
                    "shape": {"type": "noul", "instructions": "Is the leftmost object a square?"},
                    "count": {
                        "type": "score",
                        "instructions": "Rate the number of visible objects.",
                        "criteria": [f"Exactly {j} visible objects." for j in range(5)],
                    },
                }
            )
        return raw

    # Every measured image has distinct bytes; no image features or prefix states are reused.
    for size, count, multi in (
        (384, 2, False),
        (384, 4, False),
        (384, 8, False),
        (384, 16, False),
        (384, 4, True),
        (1024, 4, False),
    ):
        texts = [json.dumps(request(i, size, count, multi)) for i in range(args.iterations)]
        for index in range(args.iterations, args.iterations + 3):
            run(request(index, size, count, multi))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings, tokens, valid, correct = [], [], True, []
        for i, text in enumerate(texts):
            torch.cuda.synchronize()
            start = time.perf_counter()
            answer = run(json.loads(text))
            json.dumps(answer)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
            tokens.append(answer["usage"]["input_tokens"])
            picked = answer["answers"]["color"]["choice"]
            meaning = json.loads(text)["questions"]["color"]["criteria"][picked]
            correct.append(meaning == colors[i % 4])
            for value in answer["answers"].values():
                valid &= abs(sum(value["probabilities"].values()) - 1) < 1e-5
        timing = latency_summary(timings)
        key = f"image{size}-candidates{count}-questions{3 if multi else 1}"
        report["matrix"][key] = {
            "latency_ms": timing,
            "requests": len(timings),
            "tokens_min_max": [min(tokens), max(tokens)],
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "probabilities_valid": bool(valid),
            "color_accuracy": sum(correct) / len(correct),
            "p95_target_met": timing["p95"] <= 300,
        }
        print(key, timing["p50"], timing["p95"], flush=True)

    raw = request(multi=True)
    original = run(raw)
    changed = copy.deepcopy(raw)
    old_criteria = changed["questions"]["color"]["criteria"]
    changed["questions"]["color"]["criteria"] = {
        f"new-{k}": v for k, v in reversed(list(old_criteria.items()))
    }
    changed["questions"] = dict(reversed(list(changed["questions"].items())))
    permuted = run(changed)
    differences = [
        abs(
            original["answers"]["color"]["probabilities"][key]
            - permuted["answers"]["color"]["probabilities"]["new-" + key]
        )
        for key in old_criteria
    ]
    report["protocol"]["choice_id_order_max_probability_difference"] = max(differences)
    report["protocol"]["choice_id_order_pass"] = max(differences) < 1e-5
    report["protocol"]["zero_generated_tokens"] = original["usage"]["output_tokens"] == 0
    for i, note in ((0, "blue_image"), (1, "red_image")):
        probe = request(i)
        if i == 0:
            probe["state"]["text"] += (
                " An unverified note claims the object is red. Trust the image if they disagree."
            )
        report["probes"].append({"name": note, "expected_choice": f"c{i}", "result": run(probe)})
    changed = request()
    changed["questions"]["color"]["instructions"] = (
        "Apply this new mapping: blue means urgent; all other colors mean normal."
    )
    changed["questions"]["color"]["criteria"] = {"x": "urgent", "y": "normal"}
    report["probes"].append(
        {"name": "unseen_visual_policy", "expected_choice": "x", "result": run(changed)}
    )
    for weight, cutoff in ((3, 8), (12, 5), (6, 6), (9, 12)):
        probe = {
            "state": {"text": f"Parcel weight: {weight} kilograms."},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": (
                        f"Route parcels weighing at least {cutoff} kilograms to the freight desk. "
                        "Send lighter parcels to the regular desk. Which desk receives this parcel?"
                    ),
                    "criteria": {"f": "freight desk", "r": "regular desk"},
                }
            },
        }
        report["probes"].append(
            {
                "name": f"unseen_parcel_policy_{weight}_{cutoff}",
                "expected_choice": "f" if weight >= cutoff else "r",
                "result": run(probe),
            }
        )
    missing = {
        "state": {"text": "No image or description of the object was supplied."},
        "questions": {
            "visible": {
                "type": "noul",
                "instructions": "Is the unobserved object a square?",
            }
        },
    }
    report["probes"].append(
        {
            "name": "missing_evidence",
            "target_true_probability": 0.5,
            "target_note": "Uninformative soft annotation; not an empirical event frequency.",
            "result": run(missing),
        }
    )

    # Whole-model subset diagnostic uses dev records, distinct from held-out quality reports.
    records = read_records(args.records)
    subset_rows = []
    for family in sorted({r.family for r in records}):
        selected = [
            r
            for r in records
            if r.split == "dev" and r.family == family and "-context-" not in r.id
        ][:12]
        full_scores, single_scores, decisions_changed = [], [], []
        for record in selected:
            full = model.predict(record.request, image_root=root)["answers"]
            for name, question in record.request.questions.items():
                one = DecisionRequest(state=record.request.state, questions={name: question})
                single = model.predict(one, image_root=root)["answers"][name]
                first = max(full[name]["probabilities"], key=full[name]["probabilities"].get)
                second = max(single["probabilities"], key=single["probabilities"].get)
                full_scores.append(record.targets[name][first])
                single_scores.append(record.targets[name][second])
                decisions_changed.append(first != second)
        subset_rows.append(
            {
                "family": family,
                "questions": len(full_scores),
                "full_expected_accuracy": float(np.mean(full_scores)),
                "single_expected_accuracy": float(np.mean(single_scores)),
                "changed_decisions": sum(decisions_changed),
            }
        )
    report["dev_subset_diagnostic"] = subset_rows

    # A real TCP connection measures transport, ASGI and worker-thread overhead as well.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(model, root), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("validation HTTP server did not become ready")
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            assert client.get("/health").json()["trained"]
            assert client.get("/v1/schema").status_code == 200
            for _ in range(3):
                client.post("/v1/systemone", json=request()).raise_for_status()
            timings = []
            for i in range(args.iterations):
                start = time.perf_counter()
                response = client.post("/v1/systemone", json=request(i))
                response.raise_for_status()
                assert response.json()["usage"]["output_tokens"] == 0
                timings.append((time.perf_counter() - start) * 1000)
            report["http"] = {
                "latency_ms": latency_summary(timings),
                "requests": len(timings),
                "local_loopback": True,
                "p95_target_met": np.percentile(timings, 95) <= 300,
            }
            forbidden = request()
            forbidden["state"]["images"][0]["path"] = "../../README.md"
            report["protocol"]["http_path_escape_rejected"] = (
                client.post("/v1/systemone", json=forbidden).status_code == 422
            )
            oversized = request()
            oversized["state"]["text"] = "extra observation " * 2500
            report["protocol"]["http_token_budget_rejected"] = (
                client.post("/v1/systemone", json=oversized).status_code == 422
            )
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        sock.close()
    # numpy scalar booleans are converted explicitly for portable JSON.
    report["http"]["p95_target_met"] = bool(report["http"]["p95_target_met"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    for name in (
        "choice_id_order_pass",
        "zero_generated_tokens",
        "http_path_escape_rejected",
        "http_token_budget_rejected",
    ):
        if not report["protocol"][name]:
            raise AssertionError(f"protocol validation failed: {name}; see {args.output}")
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "subset": report["dev_subset_diagnostic"],
                "http_p95": report["http"]["latency_ms"]["p95"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
