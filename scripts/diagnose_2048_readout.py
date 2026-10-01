"""Compare text/image cell reading on balanced, rule-generated 2048 observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

from run_text_2048 import ROOT, http_json, sha256, write

from apps.playground.game2048 import board_png

VALUES = [0, 2, 4, 8, 16, 32, 64]
CRITERIA = {
    str(value): "The selected cell is empty (0)."
    if not value
    else f"The selected cell contains the tile {value}."
    for value in VALUES
}


def cases_from(source):
    pool = []
    for path in sorted((source / "random_legal").glob("*/events.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            pool.append((path.parent.name, event["revision"], event["board_before"]))
    random.Random(472319).shuffle(pool)
    cases = []
    for value in VALUES:
        selected, seen = [], set()
        for seed, revision, board in pool:
            board_key = json.dumps(board)
            positions = [(r, c) for r in range(4) for c in range(4) if board[r][c] == value]
            if positions and board_key not in seen:
                row, col = positions[len(selected) % len(positions)]
                selected.append(
                    {
                        "source_seed": seed,
                        "source_revision": revision,
                        "board": board,
                        "row": row,
                        "column": col,
                        "gold": str(value),
                    }
                )
                seen.add(board_key)
                if len(selected) == 4:
                    break
        if len(selected) != 4:
            raise ValueError(f"Insufficient distinct boards for value {value}; no inference run")
        cases.extend(selected)
    return cases


def upload(url, png):
    request = urllib.request.Request(
        url + "/api/images",
        data=png,
        headers={"Content-Type": "image/png", "X-Veyra-Playground": "1"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    status, _ = http_json(url + "/api/status")
    if status["phase"] != "ready" or status["busy"]:
        raise RuntimeError("Resident model must be ready and idle")
    cases = cases_from(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "cases.json", cases)
    write(
        args.output / "protocol.json",
        {
            "schema": "veyra.2048-cell-readout.v1",
            "values": VALUES,
            "cases_per_value": 4,
            "cases_sha256": sha256(args.output / "cases.json"),
            "script_sha256": sha256(Path(__file__)),
            "source_summary_sha256": sha256(args.source / "summary.json"),
            "checkpoint": status["checkpoint"],
            "head_sha256": sha256(ROOT / "checkpoints" / status["checkpoint"] / "head.safetensors"),
            "purpose": "Exploratory cell reading diagnostic; separate from game strategy and wins",
            "pixel_input": "512x512 classic board, no matrix or gold answer in image request",
            "model_update": False,
            "selection": "Seeded shuffle of random-control traces; 4 boards per value",
        },
    )
    records = []
    with (args.output / "events.jsonl").open("x", encoding="utf-8") as stream:
        for index, case in enumerate(cases):
            png = board_png(case["board"], 512, "classic")
            (args.output / f"{index:02}.png").write_bytes(png)
            uploaded = upload(url, png)
            question = {
                "type": "choice",
                "criteria": CRITERIA,
                "instructions": (
                    f"Read the ONE cell at row {case['row'] + 1}, column {case['column'] + 1} "
                    "of the 4 by 4 board. Row 1 is the TOP row, column 1 is the LEFTMOST column. "
                    "Choose the exact current value in that cell. An empty cell has value 0. "
                    "This is a reading task; do not move or merge tiles."
                ),
            }
            for modality in ("text", "image"):
                if modality == "text":
                    state = {
                        "text": "Current board, top row first; 0 means empty:\n"
                        + "\n".join(f"Row {i + 1}: {row}" for i, row in enumerate(case["board"]))
                    }
                else:
                    state = {
                        "text": "Read the attached 4 by 4 board image.",
                        "images": [{"path": uploaded["path"]}],
                    }
                request = {"state": state, "questions": {"cell": question}}
                started = time.perf_counter()
                response, headers = http_json(url + "/v1/systemone", request)
                answer = response["answers"]["cell"]
                record = {
                    "case": index,
                    "modality": modality,
                    "gold": case["gold"],
                    "predicted": answer["choice"],
                    "correct": answer["choice"] == case["gold"],
                    "abstained": answer["abstained"],
                    "request": request,
                    "response": response,
                    "image_sha256": hashlib.sha256(png).hexdigest(),
                    "inference_ms": float(headers["X-Veyra-Inference-Ms"]),
                    "http_ms": (time.perf_counter() - started) * 1000,
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                records.append(record)
    results = {}
    for modality in ("text", "image"):
        subset = [row for row in records if row["modality"] == modality]
        results[modality] = {
            "correct": sum(row["correct"] for row in subset),
            "total": len(subset),
            "accuracy": statistics.mean(row["correct"] for row in subset),
            "abstained": sum(row["abstained"] for row in subset),
            "model_median_ms": statistics.median(row["inference_ms"] for row in subset),
            "per_value": {
                str(value): {
                    "correct": sum(row["correct"] for row in subset if row["gold"] == str(value)),
                    "total": sum(row["gold"] == str(value) for row in subset),
                }
                for value in VALUES
            },
        }
    summary = {
        "results": results,
        "balanced_majority_baseline": 1 / len(VALUES),
        "complete": True,
        "cases": len(cases),
        "calls": len(records),
    }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
