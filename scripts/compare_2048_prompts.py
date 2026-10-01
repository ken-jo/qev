"""Compare instruction-only 2048 policies with frozen Veyra; freeze before new seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from run_text_2048 import (
    QUESTIONS,
    ROOT,
    Game2048,
    GameConfig,
    http_json,
    observation,
    sha256,
    write,
)

BASE = QUESTIONS["move"]["instructions"]
RULES = (
    "You control a 4 by 4 game of 2048. Select exactly one next arrow-key move. "
    "The objective is to create a tile with value 2048 and avoid getting stuck. "
    "Rows in the evidence are ordered top to bottom; columns are ordered left to right. "
    "Zero is empty. For LEFT or RIGHT, process each row. For UP or DOWN, process each column. "
    "First slide tiles through empty cells toward the selected edge, then merge adjacent "
    "equal values into their sum, once per original tile, then close the gaps. "
    "Unequal values cannot merge, and a newly merged tile cannot merge again in that move. "
    "Reject a direction whose slide and merges leave every cell unchanged. "
    "Among changing moves, favor equal-tile merges and positions with empty cells. "
    "Do not repeat a direction merely because it appears in recent arrow history. "
    "Read the current board anew. Return only the selected direction."
)
CORNER = RULES + (
    " Use a stable corner strategy: prefer keeping the largest tile at the bottom-left. "
    "Build a descending row along the bottom, with the next row descending in the opposite "
    "direction. Preserve that arrangement when possible, but never select an unchanged move. "
    "Avoid splitting equal-valued tiles behind larger tiles. Keep empty cells available. "
    "A changing move is preferable to preserving a corner with a move that does nothing. "
    "Use UP or RIGHT when needed to create space or enable a future merge; the corner "
    "preference is not a command to repeatedly press LEFT or DOWN."
)
EXAMPLES = CORNER + (
    " Exact rule examples for a row moving LEFT: [2,0,2,2] becomes [4,2,0,0]; "
    "[2,2,2,2] becomes [4,4,0,0], not [8,0,0,0]; [4,2,4,0] becomes [4,2,4,0] "
    "and is unchanged. For RIGHT, [2,0,2,2] becomes [0,0,2,4]. "
    "Apply these same rules to columns for vertical moves. These examples explain rules; "
    "the board to play is the current board in the evidence, not an example."
)
PROFILES = {
    "baseline": {"instructions": BASE, "history": True, "feedback": False},
    "rules": {"instructions": RULES, "history": True, "feedback": False},
    "corner": {"instructions": CORNER, "history": True, "feedback": False},
    "examples": {"instructions": EXAMPLES, "history": True, "feedback": False},
    "examples_no_history": {"instructions": EXAMPLES, "history": False, "feedback": False},
    "examples_feedback": {
        "instructions": EXAMPLES
        + (
            " Recent executed actions include the actual observed outcome. "
            "If a direction just left this same board unchanged, choose a different direction. "
            "A failed action is feedback to change the next choice, not a plan to repeat."
        ),
        "history": False,
        "feedback": True,
    },
}
DEVELOPMENT_SEEDS = [11, 29, 47, 83]
EVALUATION_SEEDS = [101, 211, 307, 401, 503, 601, 701, 809]


def request_for(game, profile):
    settings = PROFILES[profile]
    text = observation(game)
    if not settings["history"]:
        text = text.split("\nRecent arrow keys, oldest first:", 1)[0]
    if settings["feedback"] and game.events:
        text += "\nObserved results of recent executed actions (oldest first):"
        for event in game.events[-3:]:
            outcome = "changed the board" if event["moved"] else "left the board UNCHANGED"
            text += f"\n{event['direction']}: {outcome}; gained {event['score_gain']} points."
    return {
        "state": {"text": text},
        "questions": {"move": {**QUESTIONS["move"], "instructions": settings["instructions"]}},
    }


def aggregate(runs):
    events = [event for run in runs for event in run["events"]]
    times = [event["inference_ms"] for event in events]
    return {
        "games": len(runs),
        "wins": sum(run["won"] for run in runs),
        "mean_score": statistics.mean(run["score"] for run in runs),
        "mean_log2_max_tile": statistics.mean(math.log2(run["max_tile"]) for run in runs),
        "max_tile": max(run["max_tile"] for run in runs),
        "decisions": len(events),
        "valid_moves": sum(event["moved"] for event in events),
        "abstained": sum(event["abstained"] for event in events),
        "model_median_ms": statistics.median(times) if times else None,
        "directions": dict(Counter(event["direction"] for event in events)),
        "max_input_tokens": max(
            (e["response"]["usage"]["input_tokens"] for e in events), default=0
        ),
        "errors": [run["error"] for run in runs if run["error"]],
    }


def run_game(url, profile, seed, steps, folder):
    game = Game2048(GameConfig(seed=seed))
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "initial.png").write_bytes(game.image())
    started = time.perf_counter()
    reason, error = "step_budget", None
    with (folder / "events.jsonl").open("x", encoding="utf-8") as stream:
        try:
            for index in range(steps):
                payload = request_for(game, profile)
                begin = time.perf_counter()
                response, headers = http_json(url + "/v1/systemone", payload)
                http_ms = (time.perf_counter() - begin) * 1000
                answer = response["answers"]["move"]
                direction = answer.get("choice")
                if direction not in QUESTIONS["move"]["criteria"]:
                    raise ValueError("No valid direction in model response; board unchanged.")
                event = game.record(
                    direction,
                    "model",
                    abstained=bool(answer.get("abstained", False)),
                    respect_abstention=False,
                    inference_ms=float(headers["X-Veyra-Inference-Ms"]),
                    http_ms=http_ms,
                    request=payload,
                    response=response,
                )
                event.pop("frame_url")
                stream.write(json.dumps(event) + "\n")
                stream.flush()
                if (index + 1) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "profile": profile,
                                "seed": seed,
                                "step": index + 1,
                                "score": game.score,
                                "max_tile": max(map(max, game.board)),
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
                if game.state()["stats"]["no_progress_streak"] >= 4:
                    reason = "four_unchanged_actions"
                    break
        except Exception as exc:
            reason, error = "error", str(exc)
    summary = {
        "profile": profile,
        "seed": seed,
        "score": game.score,
        "max_tile": max(map(max, game.board)),
        "won": game.won,
        "steps": len(game.events),
        "stats": game.state()["stats"],
        "stop_reason": reason,
        "error": error,
        "loop_wall_ms": (time.perf_counter() - started) * 1000,
        "initial_board": game.initial_board,
        "final_board": game.board,
    }
    write(folder / "summary.json", summary)
    (folder / "final.png").write_bytes(game.image())
    print(json.dumps(summary), flush=True)
    return {**summary, "events": game.events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("Use 1..10000 decisions per game.")
    if args.phase == "evaluation" and not args.selection:
        parser.error("Evaluation requires a frozen development selection.")
    definitions_hash = hashlib.sha256(json.dumps(PROFILES, sort_keys=True).encode()).hexdigest()
    if args.phase == "development":
        profiles, seeds = list(PROFILES), DEVELOPMENT_SEEDS
        selection = None
    else:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if selection["definitions_sha256"] != definitions_hash or selection["steps"] != args.steps:
            raise ValueError("Profiles or budget changed since development; freeze again first.")
        profiles = list(dict.fromkeys(["baseline", selection["selected_profile"]]))
        seeds = EVALUATION_SEEDS
    status, _ = http_json(args.url.rstrip("/") + "/api/status")
    if status["phase"] != "ready" or status["busy"]:
        raise RuntimeError("The existing Veyra server must be ready and idle.")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "phase": args.phase,
        "seeds": seeds,
        "steps": args.steps,
        "checkpoint": status["checkpoint"],
        "runtime": status["version"],
        "device": status["device"],
        "weights_sha256": sha256(ROOT / "checkpoints" / status["checkpoint"] / "head.safetensors"),
        "script_sha256": sha256(Path(__file__)),
        "engine_sha256": sha256(ROOT / "apps/playground/game2048.py"),
        "definitions_sha256": definitions_hash,
        "profiles": {name: PROFILES[name] for name in profiles},
        "selection": selection,
        "selection_order": "wins, mean log2(max tile), mean score; earlier profile on exact ties",
        "all_four_directions_always_offered": True,
        "legal_move_filter": False,
        "solver_or_lookahead": False,
        "training_performed": False,
        "warmups_discarded": 0,
        "execution_policy": "Execute returned directions even if abstained; preserve all flags",
        "stops": ["2048", "game over", "four unchanged actions", "step budget"],
        "budget_note": (
            "The offline runner can exceed the UI session's 500-event cap. "
            "Its own 2000-step default permits enough tile mass to reach 2048."
        ),
    }
    write(args.output / "protocol.json", protocol)
    results = {}
    for profile in profiles:
        runs = []
        for seed in seeds:
            run = run_game(
                args.url.rstrip("/"), profile, seed, args.steps, args.output / profile / str(seed)
            )
            runs.append(run)
            if run["error"]:
                break
        results[profile] = {"aggregate": aggregate(runs), "runs": runs}
        write(args.output / f"{profile}.json", results[profile])
        if results[profile]["aggregate"]["errors"]:
            raise RuntimeError("Run failed; evidence preserved; no winner was selected.")
    if args.phase == "development":
        alternatives = [name for name in profiles if name != "baseline"]
        winner = max(
            alternatives,
            key=lambda name: tuple(
                results[name]["aggregate"][key]
                for key in ("wins", "mean_log2_max_tile", "mean_score")
            ),
        )
        write(
            args.output / "selection.json",
            {
                "selected_profile": winner,
                "definitions_sha256": definitions_hash,
                "steps": args.steps,
                "development_seeds": DEVELOPMENT_SEEDS,
                "evaluation_seeds": EVALUATION_SEEDS,
                "selection_order": protocol["selection_order"],
                "frozen_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    write(
        args.output / "summary.json",
        {
            "protocol": protocol,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "results": {name: result["aggregate"] for name, result in results.items()},
        },
    )


if __name__ == "__main__":
    main()
