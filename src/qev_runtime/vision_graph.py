"""Optional CUDA-graph replay of the vision tower, cached per image grid.

At batch 1 the vision tower spends much of a small image's time launching kernels and reading
grid metadata back to the host. A graph captured for one exact pixel shape and image grid
replays the same kernels on the same inputs, so its output is bitwise identical to eager
execution. Each distinct grid needs its own capture, so graphs are kept in a small LRU.
Anything the graph cannot reproduce (gradients, CPU tensors, extra outputs) runs eagerly.
"""

from collections import OrderedDict

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

# Distinct grids kept captured; a new grid beyond this evicts the least recently used one.
DEFAULT_VISION_GRAPHS = 8


class VisionGraphs:
    """Replaces ``visual.forward``; replays one captured graph per (pixel shape, grid)."""

    def __init__(self, visual, capacity=DEFAULT_VISION_GRAPHS):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("vision graph capacity must be a positive integer")
        if visual.config._attn_implementation not in {"sdpa", "eager"}:
            raise ValueError("vision graphs require sdpa or eager vision attention")
        self.visual, self.eager, self.capacity = visual, visual.forward, capacity
        self.graphs, self.pool, self.captures = OrderedDict(), None, 0

    def metadata(self, grid_thw):
        """Precompute what the tower would otherwise derive from the grid inside capture."""
        from transformers.vision_utils import (
            get_vision_attention_seqlens,
            get_vision_interpolation_indices_and_weights,
            get_vision_position_ids,
        )

        visual = self.visual
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=visual.num_grid_per_side,
            mode=visual.interpolation_mode,
            align_corners=visual.interpolation_align_corners,
            spatial_merge_size=visual.config.spatial_merge_size,
        )
        positions = get_vision_position_ids(grid_thw, visual.spatial_merge_size)
        # Attention splits by cu_seqlens.tolist() in every block; host values avoid device reads.
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw.cpu(), visual.config)
        return {
            "interp_indices": indices,
            "interp_weights": weights,
            "position_ids": positions,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }

    def capture(self, key, hidden_states, grid_thw):
        # Capture outside inference mode, as the text-stack graphs do at load: a graph kept
        # alive from inference mode leaves the CUDA generator's graph state as inference
        # tensors, and any later capture outside inference mode then fails.
        with torch.inference_mode(False), torch.no_grad():
            static = hidden_states.clone()
            grid = grid_thw.clone()
            metadata = self.metadata(grid)

            def run():
                return self.eager(static, grid_thw=grid, return_dict=True, **metadata).pooler_output

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                output = run()
        self.pool = graph.pool()
        self.captures += 1
        # The graph reads static, grid and metadata by address: they must outlive it.
        self.graphs[key] = graph, static, output, (grid, metadata)
        while len(self.graphs) > self.capacity:
            self.graphs.popitem(last=False)
        return self.graphs[key]

    def __call__(self, hidden_states, grid_thw=None, **kwargs):
        # The multimodal forward passes use_cache through; the vision tower has no cache.
        extra = {name for name, value in kwargs.items() if value is not None} - {
            "return_dict",
            "use_cache",
        }
        if (
            extra
            or grid_thw is None
            or torch.is_grad_enabled()
            or not hidden_states.is_cuda
            or kwargs.get("return_dict") is False
        ):
            return self.eager(hidden_states, grid_thw=grid_thw, **kwargs)
        key = (
            tuple(hidden_states.shape),
            hidden_states.dtype,
            tuple(map(tuple, grid_thw.tolist())),
        )
        entry = self.graphs.get(key)
        if entry is None:
            try:
                entry = self.capture(key, hidden_states, grid_thw)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                return self.eager(hidden_states, grid_thw=grid_thw, **kwargs)
        else:
            self.graphs.move_to_end(key)
        graph, static, output, _ = entry
        static.copy_(hidden_states)
        graph.replay()
        # Copy out: the next replay of any graph in the shared pool may reuse this memory.
        return BaseModelOutputWithPooling(pooler_output=output.clone())


def enable_vision_graphs(visual, capacity=DEFAULT_VISION_GRAPHS):
    """Route the vision tower through per-grid graphs captured on first use; returns the cache."""
    if not next(visual.parameters()).is_cuda:
        raise ValueError("CUDA graphs require the model on a CUDA device")
    graphs = VisionGraphs(visual, capacity)
    visual.forward = graphs
    return graphs
