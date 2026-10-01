"""Validate and run a QEV decision, preserving uncertainty for the host agent."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_response(request, response):
    if not isinstance(response, dict) or response.get("model") != "veyra-qwen3.5-2b":
        raise ValueError("The endpoint did not return the expected QEV model identity")
    backbone = response.get("backbone")
    if not isinstance(backbone, dict) or backbone.get("model_id") != "Qwen/Qwen3.5-2B":
        raise ValueError("Unexpected QEV backbone identity")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(request.questions):
        raise ValueError("Response question IDs do not match the request")
    for key, question in request.questions.items():
        answer = answers[key]
        if not isinstance(answer, dict) or answer.get("type") != question.type:
            raise ValueError("Response decision type does not match: " + key)
        expected = (
            set(question.criteria)
            if question.type == "choice"
            else {"false", "true"}
            if question.type == "noul"
            else {str(i) for i in range(len(question.criteria))}
        )
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != expected:
            raise ValueError("Response candidates do not match: " + key)
        if not all(
            type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1 for p in probs.values()
        ):
            raise ValueError("Invalid candidate probability: " + key)
        if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Candidate probabilities do not sum to one: " + key)
        confidence = answer.get("confidence")
        if (
            type(confidence) not in (float, int)
            or not math.isfinite(confidence)
            or not math.isclose(confidence, max(probs.values()), abs_tol=1e-5)
        ):
            raise ValueError("Response confidence is inconsistent: " + key)
        for field in ("abstained", "calibrated", "abstention_policy_fitted"):
            if type(answer.get(field)) is not bool:
                raise ValueError("Missing or invalid uncertainty metadata: " + key)
        if question.type == "choice":
            choice = answer.get("choice")
            if (
                not isinstance(choice, str)
                or choice not in probs
                or probs[choice] != max(probs.values())
            ):
                raise ValueError("Response choice is inconsistent: " + key)
        else:
            value = answer.get(question.type)
            expected_value = (
                probs["true"]
                if question.type == "noul"
                else sum(int(level) * probability for level, probability in probs.items())
            )
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or not math.isclose(value, expected_value, abs_tol=1e-5)
            ):
                raise ValueError("Response value is inconsistent: " + key)
    return answers


def summarize(request, response, *, min_confidence=None, review_labels=()):
    answers = validate_response(request, response)
    decisions = {}
    for key, answer in answers.items():
        reasons = []
        if answer["abstained"]:
            reasons.append("model_abstained")
        if min_confidence is not None and answer["confidence"] < min_confidence:
            reasons.append("application_confidence_floor")
        if answer["type"] == "choice" and answer["choice"] in review_labels:
            reasons.append("explicit_review_candidate")
        decisions[key] = {
            "type": answer["type"],
            "value": answer[answer["type"]],
            "status": "review" if reasons else "decided",
            "reasons": reasons,
        }
    review = any(item["status"] == "review" for item in decisions.values())
    return {
        "model_called": True,
        "status": "review" if review else "decided",
        "decisions": decisions,
        "response": response,
    }


def http_predict(endpoint, payload, timeout):
    url = urlsplit(endpoint)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Use an explicit HTTP(S) QEV endpoint without credentials or query parameters"
        )
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Earlier QEV playground servers require this explicit API-client marker.
            "X-Veyra-Playground": "1",
        },
        method="POST",
    )
    with build_opener(NoRedirect()).open(request, timeout=timeout) as stream:
        raw = stream.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("Response exceeds 1 MiB")
        return json.loads(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--backend", choices=["http", "sdk"], default="http")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/systemone")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument("--review-label", action="append", default=[])
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.min_confidence is not None and not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between zero and one")
    if args.backend == "http" and (
        args.image_root
        or args.checkpoint
        or args.cache_dir
        or args.offline
        or args.device != "auto"
    ):
        parser.error(
            "Model/cache/image-root flags apply to --backend sdk; configure the HTTP server"
        )
    try:
        import qev

        raw = args.request.read_bytes()
        if len(raw) > 1_048_576:
            raise ValueError("Request exceeds 1 MiB")
        request = qev.DecisionRequest.from_json(raw.decode("utf-8-sig"))
        if args.dry_run:
            result = {"model_called": False, "status": "validated", "request": request.model_dump()}
        else:
            if args.backend == "sdk":
                import torch

                torch.set_num_threads(4)
                model = qev.load(
                    args.checkpoint,
                    device=args.device,
                    cache_dir=args.cache_dir,
                    offline=args.offline,
                )
                response = model.predict(
                    request, image_root=(args.image_root or args.request.parent).resolve()
                )
            else:
                response = http_predict(args.endpoint, request.model_dump(), args.timeout)
            result = summarize(
                request,
                response,
                min_confidence=args.min_confidence,
                review_labels=args.review_label,
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result["status"] == "review" else 0
    except ImportError:
        print(
            "Install QEV SDK 0.2.0 in the Python environment used for this helper.", file=sys.stderr
        )
    except HTTPError as error:
        print(
            f"QEV HTTP {error.code}; check the request and selected server. No retry was made.",
            file=sys.stderr,
        )
    except (OSError, URLError, ValueError, RuntimeError) as error:
        print(f"QEV error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
