"""Training-only supervision of four existing internal positions; no inference changes."""

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from veyra.reasoning_workspace import RESULT_MARKER, condition_positions


class WorkspaceSupervisor:
    """Read existing A/B/C/D positions in the same forward used for the decision loss."""

    def __init__(self, model, seed):
        if model.reasoning_slots != 4 or model.decision_views != 1:
            raise ValueError("workspace supervision requires the existing four-position model")
        tokenizer = model.encoder.processor.tokenizer
        self.marker_ids = tokenizer.encode(RESULT_MARKER, add_special_tokens=False)
        dot = tokenizer.encode(" .", add_special_tokens=False)
        if len(dot) != 1:
            raise ValueError("the pinned tokenizer must encode each workspace dot as one token")
        self.dot_id = dot[0]
        self.enabled, self.captured = False, None
        # Initializing a separate decoder must not alter the matched decision-training RNG.
        devices = [model.encoder.device] if model.encoder.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 10000)
            self.decoder = nn.Linear(2048, 1)
        self.decoder = self.decoder.to(device=model.encoder.device, dtype=torch.float32)
        self.hook = model.encoder.model.register_forward_hook(self.capture, with_kwargs=True)

    def capture(self, _module, _args, kwargs, result):
        if not self.enabled:
            return
        if self.captured is not None:
            raise ValueError("an auxiliary graph survived into another forward")
        ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
        at = condition_positions(ids, mask, self.marker_ids).to(ids.device)
        positions = at[:, None] - torch.tensor([4, 3, 2, 1], device=ids.device)
        if (positions < 0).any():
            raise ValueError("invalid workspace position")
        if not torch.all(ids.gather(1, positions) == self.dot_id) or not torch.all(
            mask.gather(1, positions) == 1
        ):
            raise ValueError("the existing four workspace dots were not found before the marker")
        hidden = result.last_hidden_state
        states = hidden[torch.arange(hidden.shape[0], device=hidden.device)[:, None], positions]
        self.captured = states.float()

    @contextlib.contextmanager
    def observe(self, enabled):
        if self.enabled or self.captured is not None:
            raise ValueError("nested or uncleared auxiliary capture")
        self.enabled = enabled
        try:
            yield self
        finally:
            self.enabled = False
            self.captured = None

    def logits(self):
        if self.captured is None:
            raise ValueError("no internal states captured for auxiliary supervision")
        states = F.layer_norm(self.captured, (self.captured.shape[-1],))
        return self.decoder(states).squeeze(-1)

    def close(self):
        self.hook.remove()
        self.enabled = False
        self.captured = None
