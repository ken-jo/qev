"""Small, seeded 2048 environment. Only rendered pixels enter the model request."""

from __future__ import annotations

import hashlib
import io
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

DIRECTIONS = ("up", "down", "left", "right")
CHOICES = {
    "up": "Press the UP arrow: slide every tile toward the top edge.",
    "down": "Press the DOWN arrow: slide every tile toward the bottom edge.",
    "left": "Press the LEFT arrow: slide every tile toward the left edge.",
    "right": "Press the RIGHT arrow: slide every tile toward the right edge.",
}
PROMPT_VERSION = "visual-2048-v1"
MAX_EVENTS = 2500


class GameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    seed: int = Field(default=11, ge=0, le=4294967295)
    image_size: Literal[256, 384, 512, 768] = 512
    theme: Literal["classic", "contrast"] = "classic"


class StepInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: int = Field(ge=0)
    respect_abstention: bool = True
    instructions: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
        | None
    ) = None


class MoveInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: int = Field(ge=0)
    direction: Literal["up", "down", "left", "right"]


def slide(board: list[list[int]], direction: str) -> tuple[list[list[int]], int]:
    """Merge each tile once, in movement order; never spawn a tile here."""
    if direction not in DIRECTIONS:
        raise ValueError("Unknown direction")
    result = [row[:] for row in board]
    score = 0
    for lane in range(4):
        positions = (
            [(lane, index) for index in range(4)]
            if direction in ("left", "right")
            else [(index, lane) for index in range(4)]
        )
        if direction in ("right", "down"):
            positions.reverse()
        values = [board[row][column] for row, column in positions if board[row][column]]
        merged = []
        index = 0
        while index < len(values):
            value = values[index]
            if index + 1 < len(values) and value == values[index + 1]:
                value *= 2
                score += value
                index += 1
            merged.append(value)
            index += 1
        merged.extend([0] * (4 - len(merged)))
        for (row, column), value in zip(positions, merged, strict=True):
            result[row][column] = value
    return result, score


def board_png(board: list[list[int]], size: int, theme: str) -> bytes:
    """Render the exact board image shown in the playground and sent to Veyra."""
    classic = {
        2: "#eee4da",
        4: "#ede0c8",
        8: "#f2b179",
        16: "#f59563",
        32: "#f67c5f",
        64: "#f65e3b",
        128: "#edcf72",
        256: "#edcc61",
        512: "#edc850",
        1024: "#edc53f",
        2048: "#edc22e",
    }
    background, empty = ("#bbada0", "#cdc1b4") if theme == "classic" else ("#17342b", "#284a3e")
    picture = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(picture)
    gap = max(6, size // 42)
    tile = (size - 5 * gap) / 4
    for row in range(4):
        for column in range(4):
            value = board[row][column]
            left = round(gap + column * (tile + gap))
            top = round(gap + row * (tile + gap))
            right, bottom = round(left + tile), round(top + tile)
            fill = (
                (classic.get(value, "#3c3a32") if theme == "classic" else "#edf7ee")
                if value
                else empty
            )
            draw.rounded_rectangle((left, top, right, bottom), radius=max(3, size // 85), fill=fill)
            if not value:
                continue
            text = str(value)
            font_size = round(tile * (0.48 if len(text) <= 2 else 0.37 if len(text) == 3 else 0.29))
            font = None
            for name in ("C:/Windows/Fonts/segoeuib.ttf", "DejaVuSans-Bold.ttf"):
                try:
                    font = ImageFont.truetype(name, font_size)
                    break
                except OSError:
                    continue
            if font is None:
                font = ImageFont.load_default(size=font_size)
            bounds = draw.textbbox((0, 0), text, font=font)
            x = (left + right - bounds[2] + bounds[0]) / 2 - bounds[0]
            y = (top + bottom - bounds[3] + bounds[1]) / 2 - bounds[1]
            ink = "#776e65" if theme == "classic" and value <= 4 else "#f9f6f2"
            if theme == "contrast":
                ink = "#17342b"
            draw.text((x, y), text, font=font, fill=ink)
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()


@dataclass
class Game2048:
    config: GameConfig
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    board: list[list[int]] = field(default_factory=lambda: [[0] * 4 for _ in range(4)])
    score: int = 0
    revision: int = 0
    busy: bool = False
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.monotonic)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.rng = random.Random(self.config.seed)
        self.spawn()
        self.spawn()
        self.initial_board = [row[:] for row in self.board]

    def spawn(self):
        empty = [
            (row, column) for row in range(4) for column in range(4) if not self.board[row][column]
        ]
        if empty:
            row, column = self.rng.choice(empty)
            self.board[row][column] = 2 if self.rng.random() < 0.9 else 4

    @property
    def won(self):
        return max(map(max, self.board)) >= 2048

    @property
    def over(self):
        return all(slide(self.board, direction)[0] == self.board for direction in DIRECTIONS)

    def state(self):
        attempts = [event for event in self.events if event["executed"]]
        decisions = [event for event in self.events if event["source"] == "model"]
        timings = sorted(event["inference_ms"] for event in decisions)
        no_progress = 0
        for event in reversed(self.events):
            if not event["executed"] or event["moved"]:
                break
            no_progress += 1
        return {
            "id": self.id,
            "config": self.config.model_dump(),
            "revision": self.revision,
            "board": self.board,
            "score": self.score,
            "max_tile": max(map(max, self.board)),
            "won": self.won,
            "game_over": self.over,
            "busy": self.busy,
            "event_limit": MAX_EVENTS,
            "limit_reached": len(self.events) >= MAX_EVENTS,
            "frame_url": f"/api/2048/games/{self.id}/frame?revision={self.revision}",
            "stats": {
                "attempts": len(attempts),
                "valid_moves": sum(event["moved"] for event in attempts),
                "model_decisions": len(decisions),
                "abstained": sum(event["abstained"] for event in decisions),
                "no_progress_streak": no_progress,
                "model_median_ms": round(statistics.median(timings), 2) if timings else None,
                "model_total_ms": round(sum(timings), 2),
            },
            "last_event": self.events[-1] if self.events else None,
        }

    def image(self, revision: int | None = None):
        if revision is None or revision == self.revision:
            board = self.board
        elif 0 <= revision < len(self.events):
            board = self.events[revision]["board_before"]
        else:
            raise ValueError("Unknown board revision")
        return board_png(board, self.config.image_size, self.config.theme)

    def model_request(self, path: str):
        # Deliberately no board matrix, score, legal-move mask, or search advice.
        history = [event["direction"] for event in self.events if event["executed"]][-6:]
        text = (
            "Play the 2048 game shown in the attached board image. Choose the next arrow key. "
            "Tiles slide to that edge, adjacent equal tiles merge once, "
            "and a new tile then appears. "
            "Aim to reach 2048 by combining equal tiles and preserving empty spaces. "
            "Avoid a direction that leaves the board unchanged. "
            "Read the tile values and positions from the image."
        )
        if history:
            text += " Recent arrow keys, oldest first: " + ", ".join(history) + "."
        return {
            "model": "Qwen/Qwen3.5-2B",
            "state": {"text": text, "images": [{"path": path}]},
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": (
                        "Which arrow-key move is best for this board? Choose one direction."
                    ),
                    "criteria": dict(CHOICES),
                }
            },
        }

    def record(self, direction: str, source: str, *, execute: bool = True, **details):
        before = [row[:] for row in self.board]
        moved, gained = False, 0
        if execute:
            next_board, gained = slide(self.board, direction)
            moved = next_board != self.board
            if moved:
                self.board = next_board
                self.score += gained
                self.spawn()
        event = {
            "revision": self.revision,
            "source": source,
            "direction": direction,
            "executed": execute,
            "moved": moved,
            "score_gain": gained,
            "score_after": self.score,
            "board_before": before,
            "board_after": [row[:] for row in self.board],
            "frame_url": f"/api/2048/games/{self.id}/frame?revision={self.revision}",
            **details,
        }
        self.events.append(event)
        self.revision += 1
        self.touched = time.monotonic()
        return event

    def report(self, checkpoint: str):
        return {
            "schema": "veyra.visual-2048.v1",
            "checkpoint": checkpoint,
            "prompt_version": PROMPT_VERSION,
            "created_unix": self.created,
            "protocol": {
                "model_observation": "board PNG, fixed rules, last six arrow keys",
                "all_four_directions_always_offered": True,
                "board_state_sent_to_model": False,
                "legal_move_filter": False,
                "search_or_heuristic_fallback": False,
                "training_performed": False,
                "inference_timing_excludes": [
                    "PNG rendering",
                    "HTTP transport",
                    "UI refresh",
                    "model loading",
                ],
                "scope": (
                    "exploratory game demo; not a computer-use benchmark "
                    "or calibrated success probability"
                ),
            },
            "initial_board": self.initial_board,
            "final": self.state(),
            "events": self.events,
        }


def image_digest(raw: bytes):
    return hashlib.sha256(raw).hexdigest()
