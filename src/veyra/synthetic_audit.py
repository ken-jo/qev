"""Offline label audits for the original synthetic diagrams, never runtime inference."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor

from veyra.policy_data import COLORS, SHAPES


def diagram_objects(path: Path) -> list[tuple[str, str]]:
    """Independently recover separated, solid-color renderer objects from their pixels."""
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("RGB"))
    occupied = np.any(pixels != 255, axis=-1)
    columns = occupied.any(axis=0)
    transitions = np.diff(np.pad(columns.astype(int), (1, 1)))
    starts, stops = np.where(transitions == 1)[0], np.where(transitions == -1)[0]
    objects = []
    for start, stop in zip(starts, stops, strict=True):
        submask = occupied[:, start:stop]
        rows = np.flatnonzero(submask.any(axis=1))
        fraction = submask[rows[0] : rows[-1] + 1].mean()
        shape = "square" if fraction > 0.9 else "circle" if fraction > 0.65 else "triangle"
        colored = pixels[:, start:stop][submask]
        if not np.all(colored == colored[0]):
            raise ValueError("this oracle only supports separated solid-color renderer objects")
        color = next(c for c in COLORS if tuple(colored[0]) == ImageColor.getrgb(c))
        objects.append((color, shape))
    return objects


def visual_predicate(objects: list[tuple[str, str]], predicate: str) -> bool:
    count = re.fullmatch(r"the image contains at least (\d+) objects", predicate)
    if count:
        return len(objects) >= int(count.group(1))
    color = re.fullmatch(r"the (leftmost|rightmost) object's color is (\w+)", predicate)
    shape = re.fullmatch(r"the (leftmost|rightmost) object is a (\w+)", predicate)
    if not objects or not (color or shape):
        raise ValueError("unsupported visual predicate")
    match = color or shape
    side, probe = match.groups()
    if probe not in (COLORS if color else SHAPES):
        raise ValueError("unsupported renderer attribute")
    observed = objects[0] if side == "leftmost" else objects[-1]
    return observed[0 if color else 1] == probe


def numeric_predicate(expression: str, values: dict[str, int]) -> bool:
    if " OR " in expression:
        return any(numeric_predicate(part, values) for part in expression.split(" OR "))
    if " AND " in expression:
        return all(numeric_predicate(part, values) for part in expression.split(" AND "))
    expression = expression.strip("() ")
    interval = re.fullmatch(r"(\d+) <= (.+?) < (\d+)", expression)
    if interval:
        low, field, high = interval.groups()
        return int(low) <= values[field] < int(high)
    comparison = re.fullmatch(r"(.+?) (>=|<) (\d+)", expression)
    if not comparison:
        raise ValueError("unsupported numeric predicate")
    field, operator, threshold = comparison.groups()
    return values[field] >= int(threshold) if operator == ">=" else values[field] < int(threshold)
