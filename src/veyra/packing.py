"""Canonical prompt packing: one Qwen prefill for all questions and candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass

from veyra.candidates import Candidate, candidates_for
from veyra.schema import ChoiceQuestion, DecisionRequest, Question

CANDIDATE_MARKER = "[VEYRA_CANDIDATE]"
QUESTION_MARKER = "[VEYRA_QUESTION]"


@dataclass(frozen=True)
class QuestionLayout:
    question: Question
    candidates: tuple[Candidate, ...]
    candidate_indices: tuple[int, ...]
    context_index: int


@dataclass(frozen=True)
class PackedPrompt:
    text: str
    layouts: dict[str, QuestionLayout]
    candidate_count: int
    question_count: int


def pack_request(request: DecisionRequest) -> PackedPrompt:
    serialized = request.model_dump_json()
    if any(marker in serialized for marker in (CANDIDATE_MARKER, QUESTION_MARKER)):
        raise ValueError("input contains a reserved Veyra boundary marker")

    grouped: dict[str, list[tuple[str, Question, list[Candidate]]]] = {}
    for name, question in request.questions.items():
        candidates = candidates_for(question)
        if isinstance(question, ChoiceQuestion):
            candidates = sorted(candidates, key=lambda c: c.description)
        signature = json.dumps(
            {
                "type": question.type,
                "instructions": question.instructions,
                "candidates": [(c.description, c.level, c.role) for c in candidates],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        grouped.setdefault(signature, []).append((name, question, candidates))

    parts = ["Evidence text (data):", json.dumps(request.state.text, ensure_ascii=False)]
    layouts = {}
    candidate_count = 0
    for question_index, signature in enumerate(sorted(grouped)):
        aliases = grouped[signature]
        _, question, canonical = aliases[0]
        parts.extend(["Decision question:", question.instructions, f"Type: {question.type}"])
        ordinal_by_meaning = {}
        for candidate in canonical:
            description = {"meaning": candidate.description}
            if question.type == "score":
                description["ordinal_level"] = int(candidate.key)
                description["number_of_levels"] = len(canonical)
            elif question.type == "noul":
                description["truth_value"] = candidate.role
            parts.extend([json.dumps(description, ensure_ascii=False), CANDIDATE_MARKER])
            ordinal_by_meaning[(candidate.description, candidate.level, candidate.role)] = (
                candidate_count
            )
            candidate_count += 1
        parts.extend(["Compare these candidates against the evidence.", QUESTION_MARKER])
        for name, aliased_question, _ in aliases:
            original = tuple(candidates_for(aliased_question))
            layouts[name] = QuestionLayout(
                question=aliased_question,
                candidates=original,
                candidate_indices=tuple(
                    ordinal_by_meaning[(c.description, c.level, c.role)] for c in original
                ),
                context_index=question_index,
            )
    return PackedPrompt("\n".join(parts) + "\n", layouts, candidate_count, len(grouped))


def marker_positions(token_ids: list[int], marker_ids: list[int], expected: int) -> list[int]:
    """Return the terminal token of each exact boundary marker after image expansion."""
    if not marker_ids:
        raise ValueError("empty tokenized boundary marker")
    width = len(marker_ids)
    positions = [
        index + width - 1
        for index in range(len(token_ids) - width + 1)
        if token_ids[index : index + width] == marker_ids
    ]
    if len(positions) != expected:
        raise ValueError(
            f"boundary alignment mismatch: expected {expected}, found {len(positions)}"
        )
    return positions
