"""Ordered, bounded image evidence shared by both QEV encoder paths."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack, contextmanager
from pathlib import Path

from PIL import Image, ImageOps

from veyra.constants import MAX_IMAGES


def image_paths(images, image_root: Path | None = None) -> list[Path]:
    if len(images) > MAX_IMAGES:
        raise ValueError(f"a request accepts at most {MAX_IMAGES} images")
    root = image_root.resolve() if image_root is not None else None
    paths = []
    for item in images:
        path = Path(item.path)
        path = (root / path).resolve() if root is not None else path.resolve()
        if root is not None and not path.is_relative_to(root):
            raise ValueError("image path is outside the configured image root")
        if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("image must be a local file no larger than 20 MiB")
        paths.append(path)
    return paths


@contextmanager
def open_images(images, image_root: Path | None = None):
    """Release every decoded image, including when a later image or processor fails."""
    paths = image_paths(images, image_root)
    with ExitStack() as stack:
        decoded = []
        for path in paths:
            with Image.open(path) as source:
                if source.width * source.height > 40_000_000:
                    raise ValueError("source image exceeds the 40 megapixel decode limit")
                oriented = ImageOps.exif_transpose(source)
                try:
                    image = oriented.convert("RGB")
                finally:
                    if oriented is not source:
                        oriented.close()
            stack.callback(image.close)
            decoded.append(image)
        yield decoded


def image_content(count: int, text: str) -> list[dict[str, str]]:
    """Number multiple images; keep the legacy zero/one-image prompt unchanged."""
    if not 0 <= count <= MAX_IMAGES:
        raise ValueError(f"a request accepts at most {MAX_IMAGES} images")
    content = []
    for index in range(count):
        if count > 1:
            content.append({"type": "text", "text": f"Image {index + 1}:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    return content


def pixel_fingerprints(images, image_root: Path, cache: dict) -> list[str]:
    """Audit all input images with the historical decoded-RGB fingerprint format."""
    fingerprints = []
    for path in image_paths(images, image_root):
        if path not in cache:
            with Image.open(path) as source:
                if source.width * source.height > 40_000_000:
                    raise ValueError("source image exceeds the 40 megapixel decode limit")
                rgb = source.convert("RGB")
                try:
                    cache[path] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                finally:
                    rgb.close()
        fingerprints.append(cache[path])
    return fingerprints


def fingerprint_context(pixels: list[str]) -> str:
    if len(pixels) < 2:
        return str(pixels[0] if pixels else None)
    return str(tuple(pixels))
