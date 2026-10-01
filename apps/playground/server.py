"""Local Veyra playground using the frozen runtime and a resident checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import io
import ipaddress
import json
import logging
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from email.header import decode_header, make_header
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
WHEEL = ROOT / "dist/workflow-runtime/veyra-0.3.0a4-py3-none-any.whl"
if WHEEL.is_file():
    sys.path.insert(0, str(WHEEL))

from veyra import __version__  # noqa: E402
from veyra.schema import DecisionRequest  # noqa: E402

if __package__:
    from .game2048 import MAX_EVENTS, Game2048, GameConfig, MoveInput, StepInput, image_digest
else:
    from game2048 import MAX_EVENTS, Game2048, GameConfig, MoveInput, StepInput, image_digest

LOG = logging.getLogger("veyra.playground")
IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_PIXELS = 40_000_000
SESSION_BYTES = 256 * 1024 * 1024
IMAGE_LONG_EDGES = (64, 128, 192, 256, 384, 512, 768, 1024)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def resize_image(path: Path, long_edge: int) -> tuple[bytes, int, int]:
    """Create each lossless variant from the original, without cropping or upscaling."""
    with Image.open(path) as source, ImageOps.exif_transpose(source) as oriented:
        with oriented.convert("RGB") as picture:
            picture.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            picture.save(output, format="PNG")
            return output.getvalue(), picture.width, picture.height


@dataclass(frozen=True)
class TailnetAccess:
    origin: str
    authority: str
    hostname: str
    login: str
    peers: frozenset[str]

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        origin = data["origin"]
        url = urlsplit(origin)
        if (
            url.scheme != "https"
            or not url.hostname
            or not url.hostname.endswith(".ts.net")
            or url.username
            or url.password
            or url.path
            or url.query
            or url.fragment
            or origin != f"https://{url.netloc.lower()}"
            or not 1 <= (url.port or 443) <= 65535
        ):
            raise ValueError("Tailscale origin must be an exact HTTPS .ts.net origin.")
        login = data["allowed_login"]
        if not isinstance(login, str) or not login.strip():
            raise ValueError("A Tailscale account login is required.")
        peers = frozenset(str(ipaddress.ip_address(value)) for value in data["allowed_ips"])
        if not peers:
            raise ValueError("At least one exact Tailscale device address is required.")
        networks = (
            ipaddress.ip_network("100.64.0.0/10"),
            ipaddress.ip_network("fd7a:115c:a1e0::/48"),
        )
        if any(
            not any(ipaddress.ip_address(peer) in network for network in networks) for peer in peers
        ):
            raise ValueError("Only Tailscale device addresses may be allowed.")
        return cls(origin, url.netloc, url.hostname, login, peers)

    def accepts(self, request: Request) -> bool:
        # Serve overwrites these headers. The socket must remain loopback-only,
        # with Uvicorn proxy_headers=False so client still identifies the proxy.
        required = (
            "x-forwarded-host",
            "x-forwarded-proto",
            "x-forwarded-for",
            "tailscale-user-login",
        )
        if any(len(request.headers.getlist(name)) != 1 for name in required):
            return False
        if (
            request.headers["x-forwarded-host"].lower() != self.authority
            or request.headers.get("host", "").lower() != self.authority
            or request.headers["x-forwarded-proto"] != "https"
            or "tailscale-funnel-request" in request.headers
        ):
            return False
        try:
            peer = str(ipaddress.ip_address(request.headers["x-forwarded-for"]))
            login = str(make_header(decode_header(request.headers["tailscale-user-login"])))
        except (ValueError, LookupError, UnicodeError):
            return False
        return peer in self.peers and login == self.login


async def bounded_body(request: Request, limit: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise HTTPException(413, "파일 또는 요청이 허용 크기를 초과했습니다.")
        body.extend(chunk)
    return bytes(body)


class Resident:
    def __init__(self, checkpoint: Path, device: str, cache_dir: Path):
        self.checkpoint = checkpoint
        self.device = device
        self.cache_dir = cache_dir
        self.phase = "loading"
        self.error = None
        self.model = None
        self.gpu = None
        self.load_ms = None
        self.completed = 0
        self.busy = False
        self.image_policy = None
        self.smart_resize = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="veyra-inference")

    def load(self):
        try:
            import torch

            from veyra.option_model import OptionModel

            torch.set_num_threads(4)
            if not (self.checkpoint / "manifest.json").is_file():
                raise FileNotFoundError(
                    "체크포인트를 찾을 수 없습니다. --checkpoint 경로를 확인하세요."
                )
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA GPU를 찾을 수 없습니다. 현재 Python의 PyTorch 설치를 확인하세요."
                )
            start = time.perf_counter()
            self.model = OptionModel.load(
                self.checkpoint,
                device=self.device,
                cache_dir=str(self.cache_dir),
                local_files_only=True,
                merge=True,
            )
            self.load_ms = (time.perf_counter() - start) * 1000
            self.gpu = torch.cuda.get_device_name() if self.device == "cuda" else "CPU"
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

            processor = self.model.encoder.processor.image_processor
            self.smart_resize = smart_resize
            self.image_policy = {
                "min_pixels": processor.size["shortest_edge"],
                "max_pixels": processor.size["longest_edge"],
                "alignment": processor.patch_size * processor.merge_size,
                "resize_enabled": processor.do_resize,
            }
            self.phase = "ready"
            LOG.info("Resident checkpoint ready: %s on %s", self.checkpoint.name, self.gpu)
        except Exception as exc:
            self.phase = "error"
            self.error = str(exc)
            LOG.exception("Could not load the resident checkpoint")

    def image_geometry(self, width: int, height: int):
        if not self.image_policy or not self.image_policy["resize_enabled"]:
            return None
        policy = self.image_policy
        resized_height, resized_width = self.smart_resize(
            height,
            width,
            factor=policy["alignment"],
            min_pixels=policy["min_pixels"],
            max_pixels=policy["max_pixels"],
        )
        return {
            "width": resized_width,
            "height": resized_height,
            "visual_tokens": (resized_width // policy["alignment"])
            * (resized_height // policy["alignment"]),
            "estimated": True,
        }

    def predict(self, payload: DecisionRequest, image_root: Path):
        import torch

        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.model.predict(payload, image_root=image_root)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return result, round((time.perf_counter() - start) * 1000, 2)


def create_playground(
    checkpoint: Path,
    device: str,
    cache_dir: Path,
    tailnet: TailnetAccess | None = None,
    bind_host: str = "127.0.0.1",
) -> FastAPI:
    direct_network = not ipaddress.ip_address(bind_host).is_loopback
    runtime = Resident(checkpoint, device, cache_dir)
    upload_parent = ROOT / ".cache/playground"
    upload_parent.mkdir(parents=True, exist_ok=True)
    upload_session = tempfile.TemporaryDirectory(prefix="images-", dir=upload_parent)
    image_root = Path(upload_session.name).resolve()
    images: dict[str, dict] = {}
    variants: dict[tuple[str, int], str] = {}
    image_lock = asyncio.Lock()
    games: dict[str, Game2048] = {}

    def get_game(game_id: str):
        game = games.get(game_id)
        if game is None:
            raise HTTPException(404, "게임 세션이 만료되었습니다. 새 게임을 시작하세요.")
        game.touched = time.monotonic()
        return game

    async def game_command(request: Request, schema):
        raw = await bounded_body(request, 32768 if schema is StepInput else 4096)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            raise HTTPException(422, json.loads(exc.json(include_input=False))) from exc

    def check_game_move(game, revision):
        if game.busy or game.revision != revision:
            raise HTTPException(
                409, "보드가 변경되었거나 판단 중입니다. 현재 보드를 다시 불러오세요."
            )
        if game.over or game.won or len(game.events) >= MAX_EVENTS:
            raise HTTPException(409, "이 게임은 종료되었습니다. 새 게임을 시작하세요.")

    async def execute_prediction(payload, transient=None):
        if runtime.phase != "ready":
            raise HTTPException(503, runtime.error or "모델을 메모리에 준비하고 있습니다.")
        if runtime.busy:
            raise HTTPException(409, "다른 요청을 처리 중입니다. 완료 후 다시 실행하세요.")
        runtime.busy = True

        def infer():
            try:
                if transient:
                    transient[0].write_bytes(transient[1])
                return runtime.predict(payload, image_root)
            finally:
                if transient:
                    transient[0].unlink(missing_ok=True)

        try:
            future = asyncio.get_running_loop().run_in_executor(runtime.executor, infer)
        except Exception:
            runtime.busy = False
            raise

        def finished(_):
            runtime.busy = False

        future.add_done_callback(finished)
        try:
            result, elapsed = await asyncio.shield(future)
            runtime.completed += 1
            return result, elapsed
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            LOG.exception("Inference failed")
            raise HTTPException(503, "추론을 완료하지 못했습니다. 서버 로그를 확인하세요.") from exc

    def describe_image(item):
        return {**item, "processing": runtime.image_geometry(item["width"], item["height"])}

    def save_image(raw, extension, width, height, *, source_path=None, long_edge=None):
        if (
            len(images) >= 100
            or sum(item["bytes"] for item in images.values()) + len(raw) > SESSION_BYTES
        ):
            raise HTTPException(413, "사진 임시 공간이 가득 찼습니다. 서버를 재시작해 비워주세요.")
        name = f"{uuid.uuid4().hex}.{extension}"
        (image_root / name).write_bytes(raw)
        images[name] = {
            "path": name,
            "url": f"/api/images/{name}",
            "width": width,
            "height": height,
            "bytes": len(raw),
            "source_path": source_path or name,
            "long_edge": long_edge,
        }
        return images[name]

    @asynccontextmanager
    async def lifespan(app):
        load_future = asyncio.get_running_loop().run_in_executor(runtime.executor, runtime.load)
        yield
        await asyncio.shield(load_future)
        await run_in_threadpool(runtime.executor.shutdown, wait=True)
        upload_session.cleanup()

    app = FastAPI(
        title="QEV Playground",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    allowed_hosts = ["127.0.0.1", "localhost", "[::1]"]
    if tailnet:
        allowed_hosts.append(tailnet.hostname)
    if not direct_network:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts, www_redirect=False)

    @app.middleware("http")
    async def local_browser_requests(request: Request, call_next):
        local_client = request.client and request.client.host in {"127.0.0.1", "::1"}
        if not local_client and not direct_network:
            return JSONResponse(
                {"detail": "로컬 또는 Tailscale 경로로 접속하세요."}, status_code=403
            )
        forwarded = any(name.startswith(("x-forwarded-", "tailscale-")) for name in request.headers)
        if request.url.path == "/api/shutdown" and (not local_client or forwarded):
            return JSONResponse({"detail": "모델 서버에서 직접 종료하세요."}, status_code=403)
        if forwarded:
            if not local_client or not tailnet or not tailnet.accepts(request):
                return JSONResponse(
                    {"detail": "허용된 Tailscale 장치가 아닙니다."}, status_code=403
                )
            expected = tailnet.origin
        else:
            if request.url.hostname not in LOCAL_HOSTS:
                try:
                    if not direct_network:
                        raise ValueError("Local access only")
                    ipaddress.ip_address(request.url.hostname or "")
                except ValueError:
                    return JSONResponse({"detail": "서버 IP 주소로 접속하세요."}, status_code=403)
            expected = f"http://{request.headers.get('host', '')}"
        if request.method not in {"GET", "HEAD"}:
            origin = request.headers.get("origin")
            if origin and origin != expected:
                return JSONResponse(
                    {"detail": "접속한 플레이그라운드와 같은 주소에서 요청하세요."}, status_code=403
                )
            if request.headers.get("x-veyra-playground") != "1":
                return JSONResponse(
                    {"detail": "X-Veyra-Playground: 1 헤더가 필요합니다."}, status_code=403
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
        )
        return response

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/2048")
    async def game_page():
        return FileResponse(STATIC / "game2048.html")

    @app.get("/api/status")
    async def status():
        return {
            "app": "veyra-playground",
            "phase": runtime.phase,
            "busy": runtime.busy,
            "error": runtime.error,
            "checkpoint": checkpoint.name,
            "version": __version__,
            "device": runtime.gpu or device,
            "load_ms": runtime.load_ms,
            "completed": runtime.completed,
            "runtime_source": "bundled-wheel" if WHEEL.is_file() else "installed-package",
            "bind_host": bind_host,
            "remote_url": tailnet.origin if tailnet else None,
            "image_policy": runtime.image_policy,
            "image_long_edges": IMAGE_LONG_EDGES,
            "limits": {"images": 1, "questions": 4, "candidates": 16, "processed_tokens": 2048},
        }

    @app.get("/api/schema")
    async def schema():
        return DecisionRequest.model_json_schema()

    @app.post("/api/shutdown")
    async def shutdown():
        runtime.phase = "stopping"
        app.state.server.should_exit = True
        return {"status": "stopping"}

    @app.post("/api/2048/games")
    async def game_create(request: Request):
        config = await game_command(request, GameConfig)
        expired = [
            key
            for key, game in games.items()
            if not game.busy and time.monotonic() - game.touched > 7200
        ]
        for key in expired:
            del games[key]
        if len(games) >= 16:
            raise HTTPException(429, "게임 세션이 가득 찼습니다. 사용하지 않는 게임을 닫아주세요.")
        game = Game2048(config)
        games[game.id] = game
        return game.state()

    @app.get("/api/2048/games/{game_id}")
    async def game_state(game_id: str):
        return get_game(game_id).state()

    @app.delete("/api/2048/games/{game_id}")
    async def game_delete(game_id: str):
        game = get_game(game_id)
        if game.busy:
            raise HTTPException(409, "진행 중인 판단이 끝난 뒤 게임을 닫아주세요.")
        del games[game_id]
        return {"deleted": True}

    @app.get("/api/2048/games/{game_id}/frame")
    async def game_frame(game_id: str, revision: int | None = None):
        game = get_game(game_id)
        try:
            raw = game.image(revision)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(raw, media_type="image/png")

    @app.get("/api/2048/games/{game_id}/report")
    async def game_report(game_id: str):
        return get_game(game_id).report(checkpoint.name)

    @app.post("/api/2048/games/{game_id}/move")
    async def game_move(game_id: str, request: Request):
        command = await game_command(request, MoveInput)
        game = get_game(game_id)
        check_game_move(game, command.revision)
        game.record(command.direction, "manual")
        return game.state()

    @app.post("/api/2048/games/{game_id}/step")
    async def game_step(game_id: str, request: Request):
        started = time.perf_counter()
        command = await game_command(request, StepInput)
        game = get_game(game_id)
        check_game_move(game, command.revision)
        game.busy = True
        try:
            png = game.image()
            name = f"{uuid.uuid4().hex}.png"
            request_data = game.model_request(name)
            if command.instructions is not None:
                request_data["questions"]["move"]["instructions"] = command.instructions
            payload = DecisionRequest.model_validate(request_data)
            result, elapsed = await execute_prediction(payload, (image_root / name, png))
            answer = result["answers"]["move"]
            direction = answer.get("choice")
            probabilities = answer.get("probabilities", {})
            abstained = bool(answer.get("abstained", False))
            if direction is None and abstained and probabilities:
                direction = max(probabilities, key=probabilities.get)
            if direction not in ("up", "down", "left", "right"):
                raise HTTPException(
                    502, "모델 응답에 유효한 방향이 없습니다. 보드를 그대로 유지합니다."
                )
            event = game.record(
                direction,
                "model",
                execute=not (abstained and command.respect_abstention),
                abstained=abstained,
                respect_abstention=command.respect_abstention,
                inference_ms=elapsed,
                image_sha256=image_digest(png),
                request=request_data,
                response=result,
            )
            event["server_step_ms"] = round((time.perf_counter() - started) * 1000, 2)
        finally:
            game.busy = False
        return game.state()

    @app.post("/api/images")
    async def upload(request: Request):
        raw = await bounded_body(request, IMAGE_BYTES)
        if not raw:
            raise HTTPException(422, "빈 사진 파일입니다.")
        try:
            with Image.open(io.BytesIO(raw)) as picture:
                width, height = picture.size
                extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(picture.format)
                if extension is None or getattr(picture, "n_frames", 1) != 1:
                    raise ValueError("한 장의 JPG, PNG 또는 WebP 사진을 선택하세요.")
                if width * height > IMAGE_PIXELS:
                    raise ValueError("사진 크기는 4,000만 픽셀 이하여야 합니다.")
                if max(width, height) / min(width, height) > 200:
                    raise ValueError("사진의 가로·세로 비율은 200배 이하여야 합니다.")
                picture.verify()
            with Image.open(io.BytesIO(raw)) as metadata:
                if metadata.getexif().get(274) in (5, 6, 7, 8):
                    width, height = height, width
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, str(exc)) from exc
        async with image_lock:
            return describe_image(save_image(raw, extension, width, height))

    @app.get("/api/images/{name}/info")
    async def image_info(name: str):
        if name not in images:
            raise HTTPException(404, "현재 서버 세션에 없는 사진입니다. 다시 올려주세요.")
        return describe_image(images[name])

    @app.post("/api/images/{name}/resize")
    async def image_resize(name: str, request: Request):
        if name not in images:
            raise HTTPException(404, "현재 서버 세션에 없는 사진입니다. 다시 올려주세요.")
        body = await bounded_body(request, 1024)
        try:
            payload = json.loads(body)
            edge = payload.get("long_edge") if isinstance(payload, dict) else None
            if type(edge) is not int or edge not in IMAGE_LONG_EDGES:
                raise ValueError("지원하는 긴 변 크기를 선택하세요.")
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        source = images[images[name]["source_path"]]
        if edge >= max(source["width"], source["height"]):
            return describe_image(source)
        key = (source["path"], edge)
        async with image_lock:
            if key not in variants:
                try:
                    raw, width, height = await run_in_threadpool(
                        resize_image, image_root / source["path"], edge
                    )
                except (OSError, ValueError) as exc:
                    raise HTTPException(422, f"사진을 축소하지 못했습니다: {exc}") from exc
                variant = save_image(
                    raw, "png", width, height, source_path=source["path"], long_edge=edge
                )
                variants[key] = variant["path"]
            return describe_image(images[variants[key]])

    @app.get("/api/images/{name}")
    async def image_file(name: str):
        if name not in images:
            raise HTTPException(404, "현재 서버 세션에 없는 사진입니다. 다시 올려주세요.")
        return FileResponse(image_root / name)

    @app.post("/v1/systemone")
    async def predict(request: Request):
        if runtime.phase != "ready":
            raise HTTPException(503, runtime.error or "모델을 메모리에 준비하고 있습니다.")
        if runtime.busy:
            raise HTTPException(409, "다른 요청을 처리 중입니다. 완료 후 다시 실행하세요.")
        body = await bounded_body(request, 1024 * 1024)
        try:
            payload = DecisionRequest.from_json(body.decode("utf-8"))
            for item in payload.state.images:
                if item.path not in images:
                    raise ValueError("현재 플레이그라운드에 올린 사진만 사용할 수 있습니다.")
        except ValidationError as exc:
            raise HTTPException(422, json.loads(exc.json(include_input=False))) from exc
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        # The same single worker serializes playground and game requests, even on disconnect.
        result, elapsed = await execute_prediction(payload)
        return JSONResponse(
            result,
            headers={
                "X-Veyra-Inference-Ms": str(elapsed),
                "X-Veyra-Request-Id": uuid.uuid4().hex,
                "Server-Timing": f"inference;dur={elapsed}",
            },
        )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints/qev"
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/huggingface")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--host", default="127.0.0.1", help="Listen address; use 0.0.0.0 for direct IP access"
    )
    parser.add_argument("--tailscale-config", type=Path)
    args = parser.parse_args()
    bind_host = str(ipaddress.ip_address(args.host))
    config_path = args.tailscale_config or ROOT / f".cache/playground/tailscale-{args.port}.json"
    tailnet = (
        TailnetAccess.load(config_path)
        if ipaddress.ip_address(bind_host).is_loopback
        and (args.tailscale_config or config_path.is_file())
        else None
    )
    app = create_playground(
        args.checkpoint.resolve(), args.device, args.cache_dir.resolve(), tailnet, bind_host
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host=bind_host, port=args.port, access_log=False, proxy_headers=False)
    )
    app.state.server = server
    server.run()


if __name__ == "__main__":
    main()
