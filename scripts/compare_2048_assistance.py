"""Evaluate frozen Veyra with explicit, deterministic 2048 engine assistance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from compare_2048_prompts import CORNER, request_for
from run_text_2048 import ROOT, Game2048, GameConfig, http_json, sha256, write

from apps.playground.game2048 import CHOICES, DIRECTIONS, slide

DEVELOPMENT_SEEDS = [11, 29, 47, 83]
EVALUATION_SEEDS = [1009, 2003, 3001, 4001, 5003, 6007, 7001, 8009]
PROFILES = (
    "unassisted",
    "image_detailed",
    "legal_only",
    "next_board",
    "next_board_score",
    "random_legal",
)
INSTRUCTIONS = CORNER + (
    " The application has excluded directions that leave this board unchanged. "
    "Every supplied candidate is legal. If candidate descriptions include a resulting board, "
    "it is the exact result of ONE move BEFORE the random new tile appears. "
    "Use those results to compare future merge opportunities and preserve space. "
    "Any merge points are immediate game points, not a prediction of winning. "
    "Do not maximize immediate points at the expense of a playable board. "
    "The next random tile's location and value are unknown."
)
PROFILE_DETAILS = {
    "unassisted": "Previously selected corner instructions; all four directions offered",
    "image_detailed": "512px board PNG and corner instructions; all four directions; no matrix",
    "legal_only": "Current board and only legal arrow candidates",
    "next_board": "Legal candidates plus pre-spawn result boards and empty-cell counts",
    "next_board_score": "next_board plus each move's immediate merge points",
    "random_legal": "No model; uniformly sample a legal move using an independent RNG",
}


def preview_moves(board):
    """Do not touch a game's RNG, inspect its seed, spawn tiles, rank, or search."""
    result = {}
    for direction in DIRECTIONS:
        after, points = slide(board, direction)
        result[direction] = {
            "legal": after != board,
            "board_before_spawn": after,
            "merge_points": points,
            "empty_cells_before_spawn": sum(value == 0 for row in after for value in row),
        }
    return result


def assisted_request(board, profile, previews=None):
    if profile not in ("legal_only", "next_board", "next_board_score"):
        raise ValueError("Expected a model assistance profile")
    previews = preview_moves(board) if previews is None else previews
    legal = [direction for direction in DIRECTIONS if previews[direction]["legal"]]
    # The public choice contract requires at least two candidates. A forced move is
    # counted as an engine action, never reported as a successful model decision.
    if len(legal) < 2:
        return None
    text = "Current 2048 board. Rows: top to bottom; columns: left to right. Zero is empty.\n"
    text += "\n".join(f"Row {i + 1}: {row}" for i, row in enumerate(board))
    criteria = {}
    for direction in legal:
        preview = previews[direction]
        description = CHOICES[direction]
        if profile != "legal_only":
            description += "\nResult before random tile spawn (top row first):\n"
            description += "\n".join(str(row) for row in preview["board_before_spawn"])
            description += f"\nEmpty cells before spawn: {preview['empty_cells_before_spawn']}."
        if profile == "next_board_score":
            description += f"\nImmediate merge points: {preview['merge_points']}."
        criteria[direction] = description
    return {
        "state": {"text": text},
        "questions": {
            "move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}
        },
    }


def aggregate(runs):
    events = [event for run in runs for event in run["events"]]
    decisions = [event for event in events if event["source"] == "model"]
    times = [event["inference_ms"] for event in decisions]
    http_times = [event["http_ms"] for event in decisions]
    return {
        "games": len(runs),
        "wins": sum(run["won"] for run in runs),
        "mean_score": statistics.mean(run["score"] for run in runs),
        "mean_log2_max_tile": statistics.mean(math.log2(run["max_tile"]) for run in runs),
        "max_tile": max(run["max_tile"] for run in runs),
        "tile_counts": dict(Counter(str(run["max_tile"]) for run in runs)),
        "actions": len(events),
        "model_decisions": len(decisions),
        "forced_engine_moves": sum(event["source"] == "forced_engine" for event in events),
        "valid_moves": sum(event["moved"] for event in events),
        "invalid_moves": sum(not event["moved"] for event in events),
        "model_abstentions_executed": sum(event["abstained"] for event in decisions),
        "model_median_ms": statistics.median(times) if times else None,
        "http_median_ms": statistics.median(http_times) if http_times else None,
        "mean_game_wall_ms": statistics.mean(run["wall_ms"] for run in runs),
        "directions": dict(Counter(event["direction"] for event in events)),
        "stop_reasons": dict(Counter(run["stop_reason"] for run in runs)),
        "max_input_tokens": max(
            (event["response"]["usage"]["input_tokens"] for event in decisions), default=0
        ),
        "errors": [run["error"] for run in runs if run["error"]],
    }


def run_game(url, profile, seed, steps, folder):
    folder.mkdir(parents=True, exist_ok=False)
    game = Game2048(GameConfig(seed=seed))
    # Never consume the game's tile-spawning random stream to choose actions.
    policy_rng = random.Random(f"veyra-2048-random-legal-v1:{seed}")
    (folder / "initial.png").write_bytes(game.image())
    started = time.perf_counter()
    no_progress, reason, error = 0, "step_budget", None
    remote_game = None
    if profile == "image_detailed":
        remote_game, _ = http_json(
            url + "/api/2048/games", {"seed": seed, "image_size": 512, "theme": "classic"}
        )
        if remote_game["board"] != game.board or remote_game["event_limit"] < steps:
            raise RuntimeError("Server engine/config/budget differs from local experiment")
        (folder / "frames").mkdir()
    unassisted = profile in ("unassisted", "image_detailed")
    with (folder / "events.jsonl").open("x", encoding="utf-8") as stream:
        try:
            for index in range(steps):
                step_started = time.perf_counter()
                previews = preview_moves(game.board)
                legal = [direction for direction in DIRECTIONS if previews[direction]["legal"]]
                if not legal:
                    reason = "game_over"
                    break
                payload, response = None, None
                model_ms, http_ms = 0.0, 0.0
                abstained = False
                if profile == "random_legal":
                    source, direction = "random_control", policy_rng.choice(legal)
                elif not unassisted and len(legal) == 1:
                    source, direction = "forced_engine", legal[0]
                elif profile == "image_detailed":
                    png = game.image()
                    (folder / "frames" / f"{index:04}.png").write_bytes(png)
                    begin = time.perf_counter()
                    remote_game, _ = http_json(
                        url + f"/api/2048/games/{remote_game['id']}/step",
                        {
                            "revision": remote_game["revision"],
                            "respect_abstention": False,
                            "instructions": CORNER,
                        },
                    )
                    http_ms = (time.perf_counter() - begin) * 1000
                    remote_event = remote_game["last_event"]
                    payload, response = remote_event["request"], remote_event["response"]
                    model_ms = remote_event["inference_ms"]
                    source, direction = "model", remote_event["direction"]
                    abstained = remote_event["abstained"]
                    if remote_event["image_sha256"] != hashlib.sha256(png).hexdigest():
                        raise AssertionError("Local rendered board differs from actual model image")
                    if payload["questions"]["move"]["instructions"] != CORNER:
                        raise AssertionError("Server did not apply the detailed instructions")
                else:
                    payload = (
                        request_for(game, "corner")
                        if profile == "unassisted"
                        else assisted_request(game.board, profile, previews)
                    )
                    begin = time.perf_counter()
                    response, headers = http_json(url + "/v1/systemone", payload)
                    http_ms = (time.perf_counter() - begin) * 1000
                    model_ms = float(headers["X-Veyra-Inference-Ms"])
                    answer = response["answers"]["move"]
                    source, direction = "model", answer.get("choice")
                    abstained = bool(answer.get("abstained", False))
                    if direction not in payload["questions"]["move"]["criteria"]:
                        raise ValueError("Model returned an unoffered choice; nothing executed")
                before = [row[:] for row in game.board]
                event = game.record(
                    direction,
                    source,
                    abstained=abstained,
                    respect_abstention=False,
                    inference_ms=model_ms,
                    http_ms=http_ms,
                    request=payload,
                    response=response,
                    preview_moves=previews,
                    legal_directions=legal,
                )
                event.pop("frame_url")
                if profile == "image_detailed":
                    event["image_sha256"] = hashlib.sha256(png).hexdigest()
                    event["image_file"] = f"frames/{index:04}.png"
                    if game.board != remote_game["board"] or game.score != remote_game["score"]:
                        raise AssertionError("Local replay differs from executed server game")
                event["step_wall_ms"] = (time.perf_counter() - step_started) * 1000
                stream.write(json.dumps(event) + "\n")
                stream.flush()
                chosen = previews[direction]
                if (
                    event["moved"] != chosen["legal"]
                    or event["score_gain"] != chosen["merge_points"]
                ):
                    raise AssertionError("Preview and engine execution disagree")
                if not unassisted and not event["moved"]:
                    raise AssertionError("Assisted runner executed an illegal move")
                preview_board = chosen["board_before_spawn"]
                spawned = [
                    (row, col)
                    for row in range(4)
                    for col in range(4)
                    if preview_board[row][col] != game.board[row][col]
                ]
                if event["moved"]:
                    if len(spawned) != 1:
                        raise AssertionError("Expected exactly one random new tile")
                    row, col = spawned[0]
                    if preview_board[row][col] != 0 or game.board[row][col] not in (2, 4):
                        raise AssertionError("Spawn mutated an existing tile")
                elif game.board != before:
                    raise AssertionError("An unchanged move altered the board")
                no_progress = 0 if event["moved"] else no_progress + 1
                if (index + 1) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "profile": profile,
                                "seed": seed,
                                "step": index + 1,
                                "score": game.score,
                                "tile": max(map(max, game.board)),
                            }
                        ),
                        flush=True,
                    )
                if game.won:
                    reason = "won"
                    break
                if game.over:
                    reason = "game_over"
                    break
                if no_progress >= 4:
                    reason = "four_unchanged_actions"
                    break
        except Exception as exc:
            reason, error = "error", f"{type(exc).__name__}: {exc}"
    summary = {
        "profile": profile,
        "seed": seed,
        "score": game.score,
        "max_tile": max(map(max, game.board)),
        "won": game.won,
        "steps": len(game.events),
        "stop_reason": reason,
        "error": error,
        "wall_ms": (time.perf_counter() - started) * 1000,
        "initial_board": game.initial_board,
        "final_board": game.board,
    }
    write(folder / "summary.json", summary)
    (folder / "final.png").write_bytes(game.image())
    print(json.dumps(summary), flush=True)
    return {**summary, "events": game.events}


def protocol_for(args, status):
    checkpoint = ROOT / "checkpoints" / status["checkpoint"]
    return {
        "schema": "veyra.assisted-2048.v1",
        "phase": args.phase,
        "seeds": DEVELOPMENT_SEEDS if args.phase == "development" else EVALUATION_SEEDS,
        "evaluation_seeds": EVALUATION_SEEDS,
        "steps": args.steps,
        "profiles": PROFILE_DETAILS,
        "instructions": INSTRUCTIONS,
        "checkpoint": status["checkpoint"],
        "runtime": status["version"],
        "runtime_source": status["runtime_source"],
        "device": status["device"],
        "checkpoint_files_sha256": {
            str(path.relative_to(checkpoint)): sha256(path)
            for path in sorted(checkpoint.rglob("*"))
            if path.is_file()
        },
        "script_sha256": sha256(Path(__file__)),
        "engine_sha256": sha256(ROOT / "apps/playground/game2048.py"),
        "server_sha256": sha256(ROOT / "apps/playground/server.py"),
        "prompt_runner_sha256": sha256(ROOT / "scripts/compare_2048_prompts.py"),
        "text_runner_sha256": sha256(ROOT / "scripts/run_text_2048.py"),
        "definitions_sha256": hashlib.sha256(
            json.dumps([PROFILE_DETAILS, INSTRUCTIONS], sort_keys=True).encode()
        ).hexdigest(),
        "all_four_directions_offered_only_for": ["unassisted", "image_detailed"],
        "image_policy": "512x512 classic PNG; original image-only observation plus CORNER question",
        "forced_move_policy": "One legal direction is executed by engine; no model request",
        "execution_policy": "Execute all returned choices, preserve model abstention flags",
        "random_control": "Uniform legal choice; separate deterministic policy RNG",
        "random_spawn": "Engine uses 90% 2, 10% 4; previews never consume its RNG",
        "previews": "Single slide/merge only, before random spawn; no hidden state/seed sent",
        "search_or_heuristic_ranking": False,
        "training_or_calibration_updates": False,
        "warmups_discarded": 0,
        "stops": ["2048 tile", "game over", "four unchanged moves", "step budget", "error"],
        "success_criteria": {
            "software": "Tests and replay pass, zero errors; assisted actions all legal",
            "task": "At least one 2048 tile verified in held-out games; report full win rate",
            "improvement": "Compare all outcomes to legal-only and random controls",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--frozen-protocol", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("Use 1..10000 actions per game")
    if args.phase == "evaluation" and not args.frozen_protocol:
        parser.error("Evaluation requires the frozen development protocol")
    status, _ = http_json(args.url.rstrip("/") + "/api/status")
    if status["phase"] != "ready" or status["busy"]:
        raise RuntimeError("The resident Veyra server must be ready and idle")
    protocol = protocol_for(args, status)
    if args.phase == "evaluation":
        frozen = json.loads(args.frozen_protocol.read_text(encoding="utf-8"))
        for key in (
            "script_sha256",
            "engine_sha256",
            "server_sha256",
            "prompt_runner_sha256",
            "text_runner_sha256",
            "definitions_sha256",
            "checkpoint_files_sha256",
            "steps",
            "evaluation_seeds",
            "runtime",
            "checkpoint",
            "device",
            "runtime_source",
        ):
            if frozen[key] != protocol[key]:
                raise ValueError(f"Frozen protocol mismatch: {key}")
        protocol["frozen_protocol_sha256"] = sha256(args.frozen_protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol["started_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "protocol.json", protocol)
    results = {}
    for profile in PROFILES:
        runs = []
        for seed in protocol["seeds"]:
            result = run_game(
                args.url.rstrip("/"), profile, seed, args.steps, args.output / profile / str(seed)
            )
            runs.append(result)
            if result["error"]:
                break
        results[profile] = {
            "aggregate": aggregate(runs),
            "runs": [{k: v for k, v in run.items() if k != "events"} for run in runs],
        }
        write(
            args.output / "summary.json",
            {
                "protocol": protocol,
                "results": results,
                "complete": len(results) == len(PROFILES)
                and not results[profile]["aggregate"]["errors"],
            },
        )
        if results[profile]["aggregate"]["errors"]:
            raise RuntimeError("An experiment failed; stop and inspect its preserved evidence")


if __name__ == "__main__":
    main()
