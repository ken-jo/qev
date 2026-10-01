"""Fixed internal compute positions; no generated answer or supplied intermediate truth."""

import torch

RESULT_MARKER = "VeyraResult:"


def workspace_suffix(slots: int) -> str:
    if type(slots) is not int or not 1 <= slots <= 32:
        raise ValueError("workspace requires 1 to 32 fixed compute positions")
    return (
        "\nEvaluate the evidence and conditions internally.\nCondition state:"
        + " ." * slots
        + "\n"
        + RESULT_MARKER
    )


def condition_positions(input_ids, attention_mask, marker_ids):
    """Use the last appended marker, so user text cannot impersonate the internal position."""
    positions = []
    for row, mask in zip(input_ids.tolist(), attention_mask.tolist(), strict=True):
        matches = [
            start
            for start in range(1, len(row) - len(marker_ids) + 1)
            if row[start : start + len(marker_ids)] == marker_ids
            and all(mask[start - 1 : start + len(marker_ids)])
        ]
        if not matches:
            raise ValueError("internal condition marker not found after tokenization")
        positions.append(matches[-1] - 1)
    return torch.tensor(positions, dtype=torch.long)
