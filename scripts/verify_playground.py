"""Exercise real text and photo decisions through the public Gradio API."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from gradio_client import Client, handle_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860/")
    parser.add_argument("--output", type=Path, default=Path("runs/verification/space-http.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "apps/hf_space"
    presets = json.loads((folder / "presets.json").read_text("utf-8"))
    client = Client(args.url, verbose=False)
    results = {}
    for key in ("support", "photo", "photo_score", "photo_truth"):
        preset = presets[key]
        question = preset["questions"][0]
        criteria = "\n".join(
            item["text"] if question["type"] == "score" else item["key"] + " | " + item["text"]
            for item in question["options"]
        )
        image = (
            handle_file(str(folder / "samples" / (preset["sampleId"] + ".jpg")))
            if "sampleId" in preset
            else None
        )
        start = time.perf_counter()
        result = client.predict(
            preset["text"],
            image,
            question["instructions"],
            question["type"],
            criteria,
            "Original",
            api_name="/predict",
        )
        request = json.loads(result[2])
        response = result[3]
        assert response["usage"]["output_tokens"] == 0
        assert len(response["answers"]) == 1
        assert bool(request["state"]["images"]) == bool(image)
        probabilities = response["answers"]["decision"]["probabilities"]
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
        assert abs(sum(probabilities.values()) - 1) < 1e-5
        results[key] = {
            "round_trip_ms": round((time.perf_counter() - start) * 1000, 2),
            "summary": result[0],
        }
        print(json.dumps({"example": key, **results[key]}), flush=True)
    report = {
        "passed": True,
        "url": args.url,
        "real_model_requests": 4,
        "results": results,
        "source_hashes": {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("app.py", "engine.py", "presets.json", "requirements.txt")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
