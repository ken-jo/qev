"""Run one frozen model on the shared release benchmark, without fitting anything."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", "utf-8")


def laya_truncation(agent, item):
    from laya.common import render_options

    q = agent._to_internal(next(iter(item["questions"].values())))
    tok = agent.tok

    def length(text):
        return len(tok(text.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"])

    instruction = length(q["t"] + " question: " + q["ins"])
    original = [1 + length(" " + option) for option in render_options(q)]
    kept = [min(49, size) for size in original]
    budget = agent.cfg.get("head_max_len", 192) - sum(kept)
    if budget < 16:
        limit = max(4, (agent.cfg.get("head_max_len", 192) - 16) // len(kept))
        kept = [min(size, limit) for size in kept]
        budget = agent.cfg.get("head_max_len", 192) - sum(kept)
    head = min(instruction, max(8, budget))
    return {
        "instructions": int(head < instruction),
        "options": int(original != kept),
        "state": int(length(item["text"]) > max(0, 1024 - head - sum(kept) - 4)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qev", "laya-base", "laya-specialist"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--laya-sdk", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        raise RuntimeError("Use python -I so inference imports come from the distribution")
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads((args.inputs / "protocol.json").read_text("utf-8"))
    if sha(args.inputs / "requests.jsonl") != protocol["requests_sha256"]:
        raise ValueError("Frozen requests changed")
    sys.path.insert(0, str(args.wheel.resolve()))
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch

    import qev
    from veyra.data import Source, TrainingRecord
    from veyra.decision_metrics import report

    items = [
        json.loads(line)
        for line in (args.inputs / "requests.jsonl").read_text("utf-8").splitlines()
    ]
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("The matched benchmark requires CUDA")
    args.output.mkdir(parents=True)
    evidence = {
        "model": args.model,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_protocol_sha256": sha(args.inputs / "protocol.json"),
        "requests_sha256": protocol["requests_sha256"],
        "wheel_sha256": sha(args.wheel),
        "evaluator_sha256": sha(Path(__file__)),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_threads": 4,
        "cuda": torch.version.cuda,
        "qev_version": qev.__version__,
        "runtime_defaults": "Released probability calibration; no new fitting",
        "latency_scope": "Resident serial SDK calls; 3 warmups per suite; no HTTP/WAN/loading",
    }
    started = time.perf_counter()
    if args.model == "qev":
        evidence["weights_sha256"] = sha(args.checkpoint / "head.safetensors")
        evidence["manifest_sha256"] = sha(args.checkpoint / "manifest.json")
        if (
            evidence["weights_sha256"]
            != "84b57aeeb987f73416ac0f796150957d1c5beb1ed373733053e46438f77a8fee"
        ):
            raise ValueError("QEV checkpoint changed")
        model = qev.QEV.load(
            args.checkpoint, cache_dir=args.cache_dir, local_files_only=True, merge=True
        )
        evidence["precision"] = "BF16 backbone; FP32 readouts"
    else:
        reference = json.loads(args.references.read_text("utf-8"))[
            0 if args.model == "laya-base" else 1
        ]
        folder = Path(reference["path"])
        for name, expected in reference["sha256"].items():
            if sha(folder / name) != expected:
                raise ValueError("LAYA reference changed: " + name)
        evidence["reference"] = reference
        sys.path.insert(0, str(args.laya_sdk.resolve()))
        import laya

        model = laya.load(str(folder), device="cuda")
        evidence["precision"] = "Stock LAYA CUDA autocast"
        evidence["laya_agent_source_sha256"] = sha(args.laya_sdk / "laya/agent.py")
        evidence["max_len"] = 1024
        evidence["head_max_len"] = model.cfg.get("head_max_len", 192)
    evidence["load_seconds"] = time.perf_counter() - started
    write(args.output / "protocol.json", evidence)
    rows, audit = [], {}
    output = args.output / "predictions.jsonl"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for suite in dict.fromkeys(item["suite"] for item in items):
            subset = [item for item in items if item["suite"] == suite]
            truncations, durations = Counter(), []

            def predict(item):
                if args.model == "qev":
                    request = qev.DecisionRequest.model_validate(
                        {"state": {"text": item["text"]}, "questions": item["questions"]}
                    )
                    return model.predict(request)
                response = model.predict(item["text"], item["questions"], max_len=1024)
                if model.device.type != "cuda":
                    raise RuntimeError("LAYA fell back to CPU")
                return response

            for item in subset[:3]:
                predict(item)
            torch.cuda.synchronize()
            for index, item in enumerate(subset):
                if args.model != "qev":
                    truncations.update(laya_truncation(model, item))
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = predict(item)
                torch.cuda.synchronize()
                elapsed = 1000 * (time.perf_counter() - start)
                durations.append(elapsed)
                name = next(iter(item["questions"]))
                answer = result["answers"][name]
                probabilities = answer.get("probabilities")
                if args.model != "qev" and item["questions"][name]["type"] == "noul":
                    probabilities = {"false": 1 - answer["noul"], "true": answer["noul"]}
                total = sum(item["targets"].values())
                record = TrainingRecord(
                    id=item["id"],
                    group_id=item["group"],
                    split="test",
                    family=item["domain"],
                    language="en",
                    source=Source(
                        id="release-comparison",
                        revision="frozen",
                        license="See source",
                        url="https://github.com/ken-jo/qev",
                    ),
                    request=qev.DecisionRequest.model_validate(
                        {"state": {"text": item["text"]}, "questions": item["questions"]}
                    ),
                    targets={name: {k: v / total for k, v in item["targets"].items()}},
                    tags=[
                        "gold_key:" + item["gold"],
                        "domain:" + item["domain"],
                        "legacy_regression" if suite == "typed_decisions" else "zero_shot",
                    ],
                )
                from veyra.decision_metrics import observation

                row = observation(
                    record, name, probabilities, abstained=answer.get("abstained", False)
                )
                # Metrics always use probability argmax, including abstained QEV decisions.
                row.update(suite=suite, latency_ms=elapsed)
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if (index + 1) % 200 == 0:
                    print(
                        json.dumps(
                            {
                                "model": args.model,
                                "suite": suite,
                                "evaluated": index + 1,
                                "total": len(subset),
                            }
                        ),
                        flush=True,
                    )
            audit[suite] = {
                "truncation_counts": dict(truncations)
                if args.model != "qev"
                else {
                    "state": 0,
                    "instructions": 0,
                    "options": 0,
                    "basis": "Frozen QEV runtime rejects over-budget inputs; all calls succeeded",
                },
                "latency_ms": {
                    "p50": float(np.percentile(durations, 50)),
                    "p95": float(np.percentile(durations, 95)),
                    "samples": len(durations),
                },
            }
    result = {
        "passed": True,
        "protocol": evidence,
        "predictions_sha256": sha(output),
        "suites": {
            suite: {
                **report([r for r in rows if r["suite"] == suite], bootstrap=True),
                **audit[suite],
            }
            for suite in audit
        },
        "all_requests_evaluated": len(rows) == len(items),
        "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
    }
    write(args.output / "evaluation.json", result)
    print(json.dumps({k: v["overall"] for k, v in result["suites"].items()}), flush=True)


if __name__ == "__main__":
    main()
