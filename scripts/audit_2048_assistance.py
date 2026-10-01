"""Replay every recorded action, check input boundaries, and verify frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

from compare_2048_assistance import (
    CORNER,
    ROOT,
    Game2048,
    GameConfig,
    aggregate,
    assisted_request,
    preview_moves,
    request_for,
    sha256,
    write,
)


def audit(phase_dir):
    report = json.loads((phase_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["complete"], "Partial study"
    protocol = report["protocol"]
    for key, relative in {
        "script_sha256": "scripts/compare_2048_assistance.py",
        "engine_sha256": "apps/playground/game2048.py",
        "server_sha256": "apps/playground/server.py",
        "prompt_runner_sha256": "scripts/compare_2048_prompts.py",
        "text_runner_sha256": "scripts/run_text_2048.py",
    }.items():
        assert protocol[key] == sha256(ROOT / relative), f"Source drift: {relative}"
    for name, expected in protocol["checkpoint_files_sha256"].items():
        assert expected == sha256(ROOT / "checkpoints" / protocol["checkpoint"] / name)
    total_games, total_events, image_frames = 0, 0, 0
    assert set(report["results"]) == set(protocol["profiles"])
    for profile, result in report["results"].items():
        assert [run["seed"] for run in result["runs"]] == protocol["seeds"]
        runs_with_events = []
        for run in result["runs"]:
            game = Game2048(GameConfig(seed=run["seed"]))
            policy_rng = random.Random(f"veyra-2048-random-legal-v1:{run['seed']}")
            assert game.initial_board == run["initial_board"]
            folder = phase_dir / profile / str(run["seed"])
            events = [
                json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()
            ]
            assert len(events) == run["steps"]
            for index, event in enumerate(events):
                assert event["revision"] == index
                assert event["board_before"] == game.board
                previews = preview_moves(game.board)
                legal = [direction for direction, value in previews.items() if value["legal"]]
                assert event["preview_moves"] == previews
                assert event["legal_directions"] == legal
                payload = event["request"]
                if profile == "random_legal":
                    assert event["source"] == "random_control"
                    assert event["direction"] == policy_rng.choice(legal)
                    assert payload is None and event["response"] is None
                elif event["source"] == "forced_engine":
                    assert profile in ("legal_only", "next_board", "next_board_score")
                    assert legal == [event["direction"]]
                    assert payload is None and event["response"] is None
                else:
                    assert event["source"] == "model"
                    if profile == "unassisted":
                        assert payload == request_for(game, "corner")
                    elif profile == "image_detailed":
                        images = payload["state"]["images"]
                        assert len(images) == 1
                        expected = game.model_request(images[0]["path"])
                        expected["questions"]["move"]["instructions"] = CORNER
                        assert payload == expected
                        raw = (folder / event["image_file"]).read_bytes()
                        assert raw == game.image()
                        assert hashlib.sha256(raw).hexdigest() == event["image_sha256"]
                        image_frames += 1
                    else:
                        assert payload == assisted_request(game.board, profile, previews)
                    answer = event["response"]["answers"]["move"]
                    assert event["direction"] == answer["choice"]
                    assert event["abstained"] == answer["abstained"]
                    assert abs(sum(answer["probabilities"].values()) - 1) < 1e-5
                    assert set(answer["probabilities"]) == set(
                        payload["questions"]["move"]["criteria"]
                    )
                    assert event["response"]["usage"]["input_tokens"] <= 2048
                replay = game.record(
                    event["direction"],
                    event["source"],
                    abstained=event["abstained"],
                    inference_ms=event["inference_ms"],
                )
                for key in (
                    "board_before",
                    "board_after",
                    "score_gain",
                    "score_after",
                    "moved",
                    "executed",
                ):
                    assert replay[key] == event[key], (
                        f"Replay mismatch: {profile}/{run['seed']}/{index}/{key}"
                    )
                if profile not in ("unassisted", "image_detailed"):
                    assert replay["moved"]
            assert run["error"] is None
            assert game.score == run["score"]
            assert game.board == run["final_board"]
            assert game.won == run["won"]
            assert max(map(max, game.board)) == run["max_tile"]
            stop = run["stop_reason"]
            assert (
                (stop == "won" and game.won)
                or (stop == "game_over" and game.over)
                or (stop == "step_budget" and len(events) == protocol["steps"])
                or (
                    stop == "four_unchanged_actions"
                    and game.state()["stats"]["no_progress_streak"] >= 4
                )
            ), "Unjustified termination"
            assert (folder / "final.png").read_bytes() == game.image()
            runs_with_events.append({**run, "events": events})
            total_games += 1
            total_events += len(events)
        assert aggregate(runs_with_events) == result["aggregate"]
    return {
        "phase": protocol["phase"],
        "games": total_games,
        "events": total_events,
        "verified_image_frames": image_frames,
        "all_checks_passed": True,
        "summary_sha256": sha256(phase_dir / "summary.json"),
    }


def audit_readout(folder, url):
    from diagnose_2048_readout import CRITERIA, VALUES

    from apps.playground.game2048 import board_png

    cases = json.loads((folder / "cases.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    report = json.loads((folder / "summary.json").read_text())
    assert sha256(folder / "cases.json") == protocol["cases_sha256"]
    assert sha256(ROOT / "scripts/diagnose_2048_readout.py") == protocol["script_sha256"]
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    assert len(cases) == 28 and len(events) == 56
    for value in VALUES:
        assert sum(case["gold"] == str(value) for case in cases) == 4
    for index, case in enumerate(cases):
        raw = (folder / f"{index:02}.png").read_bytes()
        assert raw == board_png(case["board"], 512, "classic")
        assert str(case["board"][case["row"]][case["column"]]) == case["gold"]
        subset = [event for event in events if event["case"] == index]
        assert [event["modality"] for event in subset] == ["text", "image"]
        assert subset[0]["request"]["questions"] == subset[1]["request"]["questions"]
        for event in subset:
            assert event["image_sha256"] == hashlib.sha256(raw).hexdigest()
            assert event["gold"] == case["gold"]
            answer = event["response"]["answers"]["cell"]
            assert event["predicted"] == answer["choice"]
            assert event["correct"] == (answer["choice"] == case["gold"])
            question = event["request"]["questions"]["cell"]
            assert question["criteria"] == CRITERIA
            assert f"row {case['row'] + 1}, column {case['column'] + 1}" in question["instructions"]
            state = event["request"]["state"]
            if event["modality"] == "image":
                assert state["text"] == "Read the attached 4 by 4 board image."
                assert len(state["images"]) == 1
                name = state["images"][0]["path"]
                assert "/" not in name and "\\" not in name
                request = urllib.request.Request(
                    url.rstrip("/") + "/api/images/" + name, headers={"X-Veyra-Playground": "1"}
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    assert response.read() == raw, "Uploaded image bytes changed"
            else:
                assert "images" not in state
                expected = "Current board, top row first; 0 means empty:\n" + "\n".join(
                    f"Row {i + 1}: {row}" for i, row in enumerate(case["board"])
                )
                assert state["text"] == expected
    for modality in ("text", "image"):
        subset = [event for event in events if event["modality"] == modality]
        assert sum(event["correct"] for event in subset) == report["results"][modality]["correct"]
    return {
        "cases": len(cases),
        "calls": len(events),
        "uploaded_images_verified": len(cases),
        "all_checks_passed": True,
        "summary_sha256": sha256(folder / "summary.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--readout", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    result = {
        "audit_script_sha256": sha256(Path(__file__)),
        "phases": [audit(folder) for folder in args.phase_dirs],
    }
    if args.readout:
        result["readout"] = audit_readout(args.readout, args.url)
    write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
