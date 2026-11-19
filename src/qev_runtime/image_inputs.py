"""Ordered, bounded image evidence shared by both QEV encoder paths."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from qev_runtime.constants import MAX_IMAGES


def is_array(item) -> bool:
    """True for in-memory ``ArrayImageInput`` evidence, which has no file path."""
    return hasattr(item, "pixels")


def array_numpy(pixels) -> np.ndarray:
    """An HWC uint8 array as a contiguous host numpy array."""
    if not isinstance(pixels, np.ndarray):
        pixels = pixels.detach().cpu().numpy()
    return np.ascontiguousarray(pixels)


def image_paths(images, image_root: Path | None = None) -> list[Path | None]:
    """Resolve file evidence; in-memory arrays keep their place as ``None``."""
    if len(images) > MAX_IMAGES:
        raise ValueError(f"a request accepts at most {MAX_IMAGES} images")
    root = image_root.resolve() if image_root is not None else None
    paths = []
    for item in images:
        if is_array(item):
            paths.append(None)
            continue
        path = Path(item.path)
        path = (root / path).resolve() if root is not None else path.resolve()
        if root is not None and not path.is_relative_to(root):
            raise ValueError("image path is outside the configured image root")
        if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("image must be a local file no larger than 20 MiB")
        paths.append(path)
    return paths


def draft_jpeg(source: Image.Image, max_pixels: int | None) -> None:
    """Let libjpeg decode a large JPEG at a reduced 1/2, 1/4 or 1/8 DCT scale.

    PIL only picks scales whose result still covers the requested size, so the decoded image
    keeps at least ``max_pixels`` pixels and the processor's own resize still sets the final
    size. Other formats, small JPEGs and ``max_pixels=None`` decode at full resolution.
    """
    width, height = source.size
    if max_pixels is None or source.format != "JPEG" or width * height <= 2 * max_pixels:
        return
    ratio = (max_pixels / (width * height)) ** 0.5
    source.draft("RGB", (max(1, int(width * ratio)), max(1, int(height * ratio))))


@contextmanager
def open_images(images, image_root: Path | None = None, *, max_pixels: int | None = None):
    """Release every decoded image, including when a later image or processor fails.

    ``max_pixels`` is the processor's per-image budget; it only enables the JPEG draft decode.
    """
    paths = image_paths(images, image_root)
    with ExitStack() as stack:
        decoded = []
        for item, path in zip(images, paths, strict=True):
            if path is None:
                # The same RGB pixels a lossless file of this array would decode to.
                image = Image.fromarray(array_numpy(item.pixels))
            else:
                image = decode_file(path, max_pixels)
            stack.callback(image.close)
            decoded.append(image)
        yield decoded


def decode_file(path: Path, max_pixels: int | None) -> Image.Image:
    with Image.open(path) as source:
        if source.width * source.height > 40_000_000:
            raise ValueError("source image exceeds the 40 megapixel decode limit")
        # Draft must precede EXIF transposition, which loads the pixels.
        draft_jpeg(source, max_pixels)
        oriented = ImageOps.exif_transpose(source)
        try:
            return oriented.convert("RGB")
        finally:
            if oriented is not source:
                oriented.close()


def device_images(images, image_root: Path | None, device, *, max_pixels: int | None = None):
    """Each image as a CHW ``uint8`` tensor on ``device``, for processor-side resizing.

    Files are decoded exactly as in ``open_images``; arrays skip the decode (and, when they
    are already on ``device``, the host transfer).
    """
    import torch

    tensors = []
    for item, path in zip(images, image_paths(images, image_root), strict=True):
        if path is None:
            pixels = item.pixels
            if isinstance(pixels, np.ndarray):
                pixels = torch.from_numpy(np.ascontiguousarray(pixels))
        else:
            with decode_file(path, max_pixels) as image:
                pixels = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
        tensors.append(pixels.to(device).permute(2, 0, 1))
    return tensors


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
    for item, path in zip(images, image_paths(images, image_root), strict=True):
        if path is None:
            # Matches the fingerprint of a lossless file holding the same pixels.
            pixels = array_numpy(item.pixels)
            size = (pixels.shape[1], pixels.shape[0])
            fingerprints.append(hashlib.sha256(str(size).encode() + pixels.tobytes()).hexdigest())
            continue
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
