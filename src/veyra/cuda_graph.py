"""Optional CUDA-graph replay of the text stack for short single requests.

At batch 1 a short request spends most of its text-stack time launching kernels, so replaying a
captured graph is several times faster. Inputs are right-padded to a length bucket: causal
attention and the gated-delta recurrence never look forward, so real positions are unchanged by
the padded tail. Longer inputs are compute-bound and padding them costs more than it saves, so
they, and anything else the graph cannot reproduce, run eagerly.
"""

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

# Above 768 tokens the measured gain disappears on an RTX 4060 Ti (QEV-0.8B and QEV-2B).
SHORT_BUCKETS = (256, 384, 512, 640, 768)


def pad_to_bucket(inputs_embeds, position_ids, bucket):
    """Right-pad one request's embeddings and rotary positions to ``bucket`` tokens.

    ``position_ids`` is ``(rows, 1, length)`` or None (plain text positions, one row per rotary
    section plus the text row, as the text model builds them). The tail continues after the
    largest real position so the padded tokens look like ordinary later text.
    """
    length = inputs_embeds.shape[1]
    if position_ids is None:
        position_ids = torch.arange(length, device=inputs_embeds.device).view(1, 1, -1)
        position_ids = position_ids.expand(4, 1, -1)
    embeds = inputs_embeds.new_zeros(1, bucket, inputs_embeds.shape[-1])
    embeds[:, :length] = inputs_embeds
    tail = position_ids.amax() + torch.arange(1, bucket - length + 1, device=position_ids.device)
    positions = torch.cat([position_ids, tail.view(1, 1, -1).expand(len(position_ids), 1, -1)], -1)
    return embeds, positions


class ShortInputGraphs:
    """Replaces ``language_model.forward``; replays a captured graph for short batch-1 requests."""

    def __init__(self, language_model, buckets=SHORT_BUCKETS):
        self.language_model, self.eager = language_model, language_model.forward
        self.buckets, self.graphs, self.pool = tuple(sorted(buckets)), {}, None

    def capture(self, bucket, rows=4):
        parameter = next(self.language_model.parameters())
        hidden = self.language_model.config.hidden_size
        embeds = torch.zeros(1, bucket, hidden, device=parameter.device, dtype=parameter.dtype)
        positions = torch.arange(bucket, device=parameter.device).view(1, 1, -1)
        positions = positions.expand(rows, 1, -1).contiguous()

        def run():
            return self.eager(
                inputs_embeds=embeds, position_ids=positions, use_cache=False
            ).last_hidden_state

        with torch.no_grad():
            # Warm up on a side stream so lazy kernel setup stays out of the graph.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                output = run()
        self.pool = graph.pool()
        self.graphs[bucket, rows] = graph, embeds, positions, output
        return self.graphs[bucket, rows]

    def __call__(self, inputs_embeds=None, position_ids=None, attention_mask=None, **kwargs):
        length = None if inputs_embeds is None else inputs_embeds.shape[1]
        bucket = next((b for b in self.buckets if length is not None and b >= length), None)
        extra = {name for name, value in kwargs.items() if value is not None} - {
            "use_cache",
            "cache_position",
        }
        if (
            bucket is None
            or extra
            or kwargs.get("use_cache")
            or torch.is_grad_enabled()
            or inputs_embeds.shape[0] != 1
            or not inputs_embeds.is_cuda
            or (attention_mask is not None and not bool(attention_mask.all()))
            or (position_ids is not None and position_ids.ndim != 3)
        ):
            return self.eager(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        embeds, positions = pad_to_bucket(inputs_embeds, position_ids, bucket)
        graph, static_embeds, static_positions, output = self.graphs.get(
            (bucket, len(positions))
        ) or self.capture(bucket, len(positions))
        static_embeds.copy_(embeds)
        static_positions.copy_(positions)
        graph.replay()
        # Copy out: the next replay reuses the graph's output memory.
        return BaseModelOutputWithPast(last_hidden_state=output[:, :length].clone())


def enable_short_input_graphs(language_model, buckets=SHORT_BUCKETS):
    """Capture the graphs now and route the text stack through them; returns the wrapper."""
    if not next(language_model.parameters()).is_cuda:
        raise ValueError("CUDA graphs require the model on a CUDA device")
    graphs = ShortInputGraphs(language_model, buckets)
    for bucket in graphs.buckets:
        graphs.capture(bucket)
    language_model.forward = graphs
    return graphs
