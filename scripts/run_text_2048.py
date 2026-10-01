"""Run frozen LAYA or resident Veyra on explicitly text-only 2048 observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.playground.game2048 import CHOICES, DIRECTIONS, Game2048, GameConfig  # noqa: E402

PROMPT_VERSION = "text-2048-v1"
QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": "Which arrow-key move is best for this board? Choose one direction.",
        "criteria": dict(CHOICES),
    }
}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observation(game):
    text = (
        "Play the 2048 game described below. Choose the next arrow key. "
        "Tiles slide to that edge, adjacent equal tiles merge once, "
        "and a new tile then appears. "
        "Aim to reach 2048 by combining equal tiles and preserving empty spaces. "
        "Avoid a direction that leaves the board unchanged.\n"
        "The board has 4 rows and 4 columns. Rows are listed from top to bottom, "
        "and values within each row are listed from left to right. 0 means an empty cell.\n"
    )
    text += "\n".join(
        f"Row {index + 1}: " + ", ".join(map(str, row)) for index, row in enumerate(game.board)
    )
    history = [event["direction"] for event in game.events if event["executed"]][-6:]
    if history:
        text += "\nRecent arrow keys, oldest first: " + ", ".join(history) + "."
    return text


def http_json(url, body=None):
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Veyra-Playground": "1"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read()), response.headers


def veyra_predictor(url):
    status, _ = http_json(url + "/api/status")
    if status["phase"] != "ready" or status["busy"]:
        raise RuntimeError("Veyra playground must be ready and idle.")

    def predict(text):
        started = time.perf_counter()
        result, headers = http_json(
            url + "/v1/systemone", {"state": {"text": text}, "questions": QUESTIONS}
        )
        return (
            result,
            float(headers["X-Veyra-Inference-Ms"]),
            {"http_ms": (time.perf_counter() - started) * 1000},
        )

    metadata = {
        "checkpoint": status["checkpoint"],
        "runtime": status["version"],
        "runtime_source": status["runtime_source"],
        "device": status["device"],
        "transport": "local HTTP; model time read from synchronized server header",
        "weights_sha256": sha256(ROOT / "checkpoints" / status["checkpoint"] / "head.safetensors"),
    }
    return predict, metadata


def laya_predictor(reference_index):
    os.environ.setdefault("USE_TF", "0")
    sys.path.insert(0, str(ROOT / ".cache/laya-sdk"))
    import laya
    import torch
    from laya.common import build_sequence, render_options

    references = json.loads((ROOT / "runs/foundation-v11-laya/sources.json").read_text())
    reference = references[reference_index]
    folder = Path(reference["path"])
    for name, expected in reference["sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Pinned LAYA reference changed: {name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparison.")
    torch.set_num_threads(4)
    started = time.perf_counter()
    agent = laya.load(str(folder), device="cuda")
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - started) * 1000
    if agent.device.type != "cuda":
        raise RuntimeError("LAYA fell back to CPU; release GPU memory before retrying.")
    internal = agent._to_internal(QUESTIONS["move"])

    def predict(text):
        # Check the actual SDK formatter against a generous budget before timing inference.
        # The short instruction/options are checked separately for the SDK's 48-token cap.
        for option in render_options(internal):
            if len(agent.tok(" " + option, add_special_tokens=False)["input_ids"]) > 48:
                raise ValueError("An option would be truncated.")
        sequence, markers = build_sequence(
            agent.tok, text, internal, agent.cfg["max_len"], agent.cfg["head_max_len"]
        )
        complete, full_markers = build_sequence(agent.tok, text, internal, 8192, 8192)
        if sequence != complete or markers != full_markers or len(markers) != 4:
            raise ValueError("LAYA input would be truncated; no partial input was evaluated.")
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = agent.predict(text, QUESTIONS)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if agent.device.type != "cuda":
            raise RuntimeError("LAYA changed to CPU during inference.")
        return result, elapsed, {"input_tokens": len(sequence), "input_truncated": False}

    metadata = {key: value for key, value in reference.items() if key != "path"}
    metadata.update(
        device=torch.cuda.get_device_name(),
        load_ms=load_ms,
        config=agent.cfg,
        fast_path=bool(agent._fast),
        transport="direct SDK; synchronized CUDA timing includes predict preprocessing",
    )
    return predict, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("laya-base", "laya-typed", "veyra"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47, 83])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 500 or any(not 0 <= seed <= 4294967295 for seed in args.seeds):
        parser.error("Use 1..500 steps and seeds in 0..4294967295.")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "veyra.text-2048.v1",
        "prompt_version": PROMPT_VERSION,
        "backend": args.backend,
        "seeds": args.seeds,
        "max_steps_per_seed": args.steps,
        "input": "Exact numeric board as English text from game engine; no image or OCR",
        "policy": (
            "Execute every returned choice including Veyra abstentions; preserve all flags. "
            "Do not interpret LAYA act_probability as a calibrated gate."
        ),
        "all_four_directions_always_offered": True,
        "legal_move_filter": False,
        "search_or_heuristic_fallback": False,
        "training_performed": False,
        "warmups_discarded": 0,
        "stop_conditions": ["2048 tile", "game over", "four unchanged actions", "step budget"],
        "questions": QUESTIONS,
        "script_sha256": sha256(Path(__file__)),
        "engine_sha256": sha256(ROOT / "apps/playground/game2048.py"),
        "scope": (
            "Exploratory gameplay, not an image benchmark or calibrated strategic accuracy. "
            "Closed-loop observations can diverge after different model actions."
        ),
    }
    write(args.output / "protocol.json", protocol)
    if args.backend == "veyra":
        predict, metadata = veyra_predictor(args.url.rstrip("/"))
    else:
        predict, metadata = laya_predictor(0 if args.backend == "laya-base" else 1)
    write(args.output / "model.json", metadata)
    summaries = []
    all_times = []
    for seed in args.seeds:
        game = Game2048(GameConfig(seed=seed))
        folder = args.output / f"seed-{seed}"
        folder.mkdir()
        (folder / "initial.png").write_bytes(game.image())
        reason, error = "step_budget", None
        started = time.perf_counter()
        with (folder / "events.jsonl").open("x", encoding="utf-8") as stream:
            try:
                for _ in range(args.steps):
                    text = observation(game)
                    response, elapsed, details = predict(text)
                    answer = response["answers"]["move"]
                    direction = answer.get("choice")
                    if direction not in DIRECTIONS:
                        raise ValueError("Model did not return one of the four directions.")
                    event = game.record(
                        direction,
                        "model",
                        abstained=bool(answer.get("abstained", False)),
                        respect_abstention=False,
                        inference_ms=elapsed,
                        request_text=text,
                        response=response,
                        **details,
                    )
                    event.pop("frame_url")
                    all_times.append(elapsed)
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                    stream.flush()
                    if game.won:
                        reason = "won"
                        break
                    if game.over:
                        reason = "game_over"
                        break
                    if game.state()["stats"]["no_progress_streak"] >= 4:
                        reason = "four_unchanged_actions"
                        break
            except Exception as exc:
                reason, error = "error", str(exc)
        summary = {
            "seed": seed,
            "score": game.score,
            "max_tile": max(map(max, game.board)),
            "won": game.won,
            "stats": game.state()["stats"],
            "first_inference_ms": game.events[0]["inference_ms"] if game.events else None,
            "stop_reason": reason,
            "error": error,
            "loop_wall_ms": (time.perf_counter() - started) * 1000,
            "initial_board": game.initial_board,
            "final_board": game.board,
        }
        write(folder / "summary.json", summary)
        (folder / "final.png").write_bytes(game.image())
        summaries.append(summary)
        print(json.dumps({"backend": args.backend, **summary}), flush=True)
        if error:
            break
    result = {
        "protocol": protocol,
        "model": metadata,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runs": summaries,
        "all_decisions": len(all_times),
        "model_median_ms": statistics.median(all_times) if all_times else None,
        "error": next((run["error"] for run in summaries if run["error"]), None),
    }
    write(args.output / "results.json", result)
    if result["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
