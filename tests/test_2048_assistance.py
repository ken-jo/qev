"""Validate assisted inputs without loading or changing model weights."""

import copy
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_2048_assistance import assisted_request, preview_moves  # noqa: E402
from run_text_2048 import Game2048, GameConfig  # noqa: E402

from apps.playground.game2048 import DIRECTIONS, MAX_EVENTS, StepInput, slide  # noqa: E402


@pytest.mark.parametrize(
    "row,direction,expected,points",
    [
        ([2, 0, 2, 2], "left", [4, 2, 0, 0], 4),
        ([2, 2, 2, 2], "left", [4, 4, 0, 0], 8),
        ([4, 2, 4, 0], "left", [4, 2, 4, 0], 0),
        ([2, 0, 2, 4], "right", [0, 0, 4, 4], 4),
        ([2, 2, 4, 4], "right", [0, 0, 4, 8], 12),
    ],
)
def test_known_merges_and_no_chain(row, direction, expected, points):
    board = [row, [0] * 4, [0] * 4, [0] * 4]
    result = preview_moves(board)[direction]
    assert result["board_before_spawn"][0] == expected
    assert result["merge_points"] == points
    assert result["legal"] == (row != expected)


def test_preview_has_no_side_effects_or_random_tile():
    game = Game2048(GameConfig(seed=17))
    before, rng_state = copy.deepcopy(game.board), game.rng.getstate()
    previews = preview_moves(game.board)
    assert game.board == before
    assert game.rng.getstate() == rng_state
    assert not game.events
    for result in previews.values():
        assert sum(map(sum, result["board_before_spawn"])) == sum(map(sum, before))


def test_request_information_boundaries_and_legal_candidates():
    board = [[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4]
    previews = preview_moves(board)
    for profile in ("legal_only", "next_board", "next_board_score"):
        payload = assisted_request(board, profile)
        options = payload["questions"]["move"]["criteria"]
        assert set(options) == {"down", "right"}
        assert payload["state"]["text"].startswith("Current 2048 board.")
        assert "images" not in payload["state"]
        assert "seed" not in json.dumps(payload)
        for direction, description in options.items():
            assert ("Result before random tile spawn" in description) == (profile != "legal_only")
            assert ("Immediate merge points:" in description) == (profile == "next_board_score")
            if profile != "legal_only":
                for row in previews[direction]["board_before_spawn"]:
                    assert str(row) in description


def test_forced_move_and_terminal_board_need_no_request():
    # Full distinct top three rows; only moving down changes the board.
    board = [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [0, 0, 0, 0]]
    previews = preview_moves(board)
    assert [key for key, value in previews.items() if value["legal"]] == ["down"]
    assert assisted_request(board, "next_board_score") is None
    dead = [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
    assert not any(item["legal"] for item in preview_moves(dead).values())
    assert assisted_request(dead, "legal_only") is None


def test_vertical_rules_and_mass_on_varied_boards():
    rng = random.Random(73419)
    for _ in range(250):
        board = [[rng.choice([0, 2, 4, 8, 16, 32]) for _ in range(4)] for _ in range(4)]
        before = copy.deepcopy(board)
        previews = preview_moves(board)
        transposed = list(map(list, zip(*board, strict=True)))
        for vertical, horizontal in (("up", "left"), ("down", "right")):
            expected, points = slide(transposed, horizontal)
            assert previews[vertical]["board_before_spawn"] == list(
                map(list, zip(*expected, strict=True))
            )
            assert previews[vertical]["merge_points"] == points
        for direction in DIRECTIONS:
            result = previews[direction]
            after = result["board_before_spawn"]
            assert sum(map(sum, after)) == sum(map(sum, board))
            assert result["empty_cells_before_spawn"] >= sum(v == 0 for row in board for v in row)
        assert board == before


def test_actual_games_match_previews_and_spawn_once():
    for seed in (199, 257, 389):
        game = Game2048(GameConfig(seed=seed))
        policy = random.Random(seed + 10000)
        for _ in range(500):
            previews = preview_moves(game.board)
            legal = [key for key, value in previews.items() if value["legal"]]
            if not legal:
                assert game.over
                break
            direction = policy.choice(legal)
            expected = previews[direction]
            event = game.record(direction, "verification")
            assert event["moved"]
            assert event["score_gain"] == expected["merge_points"]
            differences = [
                (r, c)
                for r in range(4)
                for c in range(4)
                if game.board[r][c] != expected["board_before_spawn"][r][c]
            ]
            assert len(differences) == 1
            row, col = differences[0]
            assert expected["board_before_spawn"][row][col] == 0
            assert game.board[row][col] in (2, 4)


def test_image_step_instruction_override_is_optional_and_bounded():
    assert StepInput(revision=0).instructions is None
    assert (
        StepInput(revision=0, instructions="  Read the image.  ").instructions == "Read the image."
    )
    for invalid in (" ", "x" * 8001):
        with pytest.raises(ValueError):
            StepInput(revision=0, instructions=invalid)
    assert MAX_EVENTS >= 2000


def test_image_step_api_uses_override_and_transient_png(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from apps.playground import server

    observed = []

    def fake_load(resident):
        resident.phase = "ready"

    def fake_predict(resident, payload, image_root):
        assert len(payload.state.images) == 1
        path = image_root / payload.state.images[0].path
        assert path.is_file()
        assert path.read_bytes().startswith(b"\x89PNG")
        assert "Row 1:" not in payload.state.text
        observed.append((path, payload.questions["move"].instructions))
        return {"answers": {"move": {"choice": "right", "abstained": False}}}, 1.0

    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server.Resident, "load", fake_load)
    monkeypatch.setattr(server.Resident, "predict", fake_predict)
    app = server.create_playground(tmp_path / "checkpoint", "cpu", tmp_path / "cache")
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        headers = {"X-Veyra-Playground": "1"}
        game = client.post("/api/2048/games", json={"seed": 11}, headers=headers).json()
        url = f"/api/2048/games/{game['id']}/step"
        response = client.post(
            url, json={"revision": 0, "instructions": "Detailed image rules."}, headers=headers
        )
        assert response.status_code == 200, response.text
        event = response.json()["last_event"]
        assert event["request"]["questions"]["move"]["instructions"] == "Detailed image rules."
        assert observed[0][1] == "Detailed image rules."
        assert not observed[0][0].exists()
        assert client.post(url, json={"revision": 0}, headers=headers).status_code == 409
        default = client.post(url, json={"revision": 1}, headers=headers)
        assert default.status_code == 200, default.text
        assert "Which arrow-key move" in observed[1][1]
        assert not observed[1][0].exists()
