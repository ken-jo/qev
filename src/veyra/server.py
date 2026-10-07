"""Local inference API. Image references are confined to one configured root."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from veyra import __version__
from veyra.constants import MODEL_ID
from veyra.model import VeyraModel
from veyra.schema import DecisionRequest


def create_app(model: VeyraModel, image_root: Path) -> FastAPI:
    if not model.trained:
        raise ValueError("the inference API requires a trained checkpoint")
    root = image_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("image_root must be a directory")
    app = FastAPI(title="Veyra", version=__version__)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ready", "trained": model.trained}

    @app.get("/v1/models")
    def models() -> dict:
        return {"models": [{"id": MODEL_ID, "decision_model": "veyra-qwen3.5-2b"}]}

    @app.get("/v1/schema")
    def schema() -> dict:
        return DecisionRequest.model_json_schema()

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> dict:
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 1024 * 1024:
                raise HTTPException(413, "request body exceeds 1 MiB")
        try:
            payload = DecisionRequest.from_systemone_json(body.decode("utf-8"))
            for image in payload.state.images:
                path = (root / image.path).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("image path is outside the configured image root")
            return await run_in_threadpool(model.predict, payload, image_root=root)
        except ValidationError as exc:
            raise HTTPException(422, json.loads(exc.json(include_input=False))) from exc
        except (ValueError, UnicodeError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
