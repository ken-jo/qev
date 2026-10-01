"""Run the playground's real image-only 2048 decision loop and preserve its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--size", type=int, choices=(256, 384, 512, 768), default=512)
    parser.add_argument("--theme", choices=("classic", "contrast"), default="classic")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--execute-abstained",
        action="store_true",
        help="Game exploration only: execute an abstained recommendation and record it.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 500:
        parser.error("--steps must be between 1 and 500")
    args.output.mkdir(parents=True, exist_ok=False)
    frames = args.output / "frames"
    frames.mkdir()

    def request(path, body=None):
        req = urllib.request.Request(
            args.url.rstrip("/") + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", "X-Veyra-Playground": "1"},
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                raw = response.read()
                return (
                    raw if response.headers.get_content_type() == "image/png" else json.loads(raw)
                )
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
            ) from exc

    status = request("/api/status")
    if status["phase"] != "ready" or status["busy"]:
        raise RuntimeError("Playground must be ready and idle before starting a run.")
    game = request(
        "/api/2048/games", {"seed": args.seed, "image_size": args.size, "theme": args.theme}
    )
    root = f"/api/2048/games/{game['id']}"
    (args.output / "session.json").write_text(
        json.dumps(
            {
                "id": game["id"],
                "url": args.url,
                "config": game["config"],
                "steps": args.steps,
                "execute_abstained": args.execute_abstained,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    timings = []
    matched_images = []
    reason = "step_budget"
    error = None
    started = time.perf_counter()
    try:
        for step in range(args.steps):
            raw = request(game["frame_url"])
            (frames / f"{game['revision']:03}.png").write_bytes(raw)
            start = time.perf_counter()
            game = request(
                root + "/step",
                {"revision": game["revision"], "respect_abstention": not args.execute_abstained},
            )
            timings.append((time.perf_counter() - start) * 1000)
            event = game["last_event"]
            matched_images.append(hashlib.sha256(raw).hexdigest() == event["image_sha256"])
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "direction": event["direction"],
                        "moved": event["moved"],
                        "abstained": event["abstained"],
                        "executed": event["executed"],
                        "score": game["score"],
                        "max_tile": game["max_tile"],
                        "model_ms": event["inference_ms"],
                    }
                ),
                flush=True,
            )
            if not matched_images[-1]:
                raise RuntimeError("Displayed frame and model input PNG hashes do not match.")
            if game["won"]:
                reason = "won"
                break
            if game["game_over"]:
                reason = "game_over"
                break
            if not event["executed"]:
                reason = "abstained"
                break
            if game["stats"]["no_progress_streak"] >= 4:
                reason = "four_unchanged_actions"
                break
    except Exception as exc:
        reason = "error"
        error = str(exc)
    elapsed = (time.perf_counter() - started) * 1000
    # A failed POST is never retried: read back the state before making any further move.
    report = request(root + "/report")
    (args.output / "final.png").write_bytes(request(root + "/frame"))
    report["runner"] = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "stop_reason": reason,
        "error": error,
        "requested_steps": args.steps,
        "execute_abstained": args.execute_abstained,
        "step_http_ms": timings,
        "http_median_ms": statistics.median(timings) if timings else None,
        "loop_wall_ms": elapsed,
        "displayed_image_matches_model": matched_images,
        "timing_note": (
            "No discarded warmups; first model call may be cold for this shape. "
            "HTTP steps exclude the separate frame download; "
            "loop wall time includes evidence downloads and file writes."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    final = report["final"]
    summary = {
        "session": final["id"],
        "config": final["config"],
        "score": final["score"],
        "max_tile": final["max_tile"],
        "won": final["won"],
        "stats": final["stats"],
        "stop_reason": reason,
        "error": error,
        "all_image_hashes_match": bool(matched_images) and all(matched_images),
        "http_median_ms": report["runner"]["http_median_ms"],
        "loop_wall_ms": elapsed,
        "url": args.url.rstrip("/") + "/2048?game=" + final["id"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
