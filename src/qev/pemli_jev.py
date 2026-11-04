#!/usr/bin/env python3
"""
PEMLI-JEV (`pemli_jev.py`): Permutation-Equivariant Matryoshka Late-Interaction Decision Engine
Vision-Embedding-based Typed Decision Architecture for `ken-jo/qev`.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DecisionType = Literal["choice", "score", "noul"]


@dataclass
class ImageInput:
    path: Optional[str] = None
    url: Optional[str] = None
    pixels: Optional[np.ndarray] = None
    resolution_mp: float = 1.6


@dataclass
class StateSpec:
    text: str = ""
    images: List[ImageInput] = field(default_factory=list)

    def state_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.text.encode("utf-8"))
        for img in self.images:
            if img.path:
                h.update(img.path.encode("utf-8"))
            h.update(f":{img.resolution_mp:.2f}".encode("utf-8"))
        return h.hexdigest()[:24]


@dataclass
class QuestionSpec:
    type: DecisionType
    instructions: str
    criteria: Dict[str, str] = field(default_factory=dict)
    levels: Optional[Dict[str, float]] = None


@dataclass
class DecisionRequest:
    state: StateSpec
    questions: Dict[str, QuestionSpec]

    @classmethod
    def model_validate(cls, payload: Dict[str, Any]) -> "DecisionRequest":
        raw_state = payload.get("state", {})
        if isinstance(raw_state, str):
            state_spec = StateSpec(text=raw_state, images=[])
        else:
            imgs = [
                img if isinstance(img, ImageInput) else ImageInput(**img)
                for img in raw_state.get("images", [])
            ]
            state_spec = StateSpec(text=raw_state.get("text", ""), images=imgs)

        questions_dict: Dict[str, QuestionSpec] = {}
        for q_id, q_val in payload.get("questions", {}).items():
            if isinstance(q_val, QuestionSpec):
                questions_dict[q_id] = q_val
            else:
                q_type = q_val.get("type", "choice")
                criteria = q_val.get("criteria", {})
                if q_type == "noul" and not criteria:
                    criteria = {"yes": "True", "no": "False"}
                questions_dict[q_id] = QuestionSpec(
                    type=q_type,
                    instructions=q_val.get("instructions", ""),
                    criteria=criteria,
                    levels=q_val.get("levels"),
                )
        return cls(state=state_spec, questions=questions_dict)


@dataclass
class CalibrationPolicy:
    temperature: float = 0.84
    cardinality_gamma: float = 0.11
    confidence_threshold: float = 0.24
    energy_threshold: float = 1.35
    evidence_sim_floor: float = 0.16
    fitted_ece: float = 0.018
    fitted_cp95_risk: float = 0.142
    fitted_coverage: float = 0.912
    status: str = "fitted"


@dataclass
class PEMLIConfig:
    backbone_id: str = "google/embeddinggemma-2"
    omit_audio_encoder: bool = True
    hidden_dim: int = 768
    mrl_dims: Tuple[int, ...] = (128, 256, 512, 768)
    active_mrl_dim: int = 512
    router_mrl_dim: int = 128
    max_visual_patches: int = 560
    top_k_patches_accurate: int = 256
    top_k_patches_fast: int = 96
    num_reasoning_slots: int = 4
    num_set_layers: int = 2
    num_heads: int = 8
    maxsim_weight: float = 0.45
    kl_teacher_weight: float = 1.0
    gold_ce_weight: float = 0.30
    contrastive_margin_weight: float = 0.25
    mrl_multislice_weight: float = 0.15
    contrastive_margin: float = 0.35
    target_max_error: float = 0.15
    min_accepted_questions: int = 20
    precision: str = "bfloat16"


if TORCH_AVAILABLE:
    class MatryoshkaLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
            nn.init.xavier_uniform_(self.weight)

        def forward(self, x: torch.Tensor, active_dim: Optional[int] = None) -> torch.Tensor:
            in_d = x.shape[-1]
            out_d = active_dim or self.weight.shape[0]
            w = self.weight[:out_d, :in_d]
            b = self.bias[:out_d] if self.bias is not None else None
            return F.linear(x, w, b)

    class QuestionGuidedPatchRouter(nn.Module):
        def __init__(self, router_dim: int = 128):
            super().__init__()
            self.router_dim = router_dim
            self.q_proj = nn.Linear(router_dim, router_dim, bias=False)
            self.p_proj = nn.Linear(router_dim, router_dim, bias=False)
            nn.init.eye_(self.q_proj.weight)
            nn.init.eye_(self.p_proj.weight)

        def forward(self, patch_tokens: torch.Tensor, question_vec: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
            B, N, D = patch_tokens.shape
            if top_k >= N:
                return patch_tokens, torch.ones(B, N, device=patch_tokens.device, dtype=patch_tokens.dtype)
            r_d = min(self.router_dim, D)
            q_fast = F.normalize(self.q_proj(question_vec[:, :r_d]), p=2, dim=-1)
            p_fast = F.normalize(self.p_proj(patch_tokens[:, :, :r_d]), p=2, dim=-1)
            importance = torch.bmm(p_fast, q_fast.unsqueeze(-1)).squeeze(-1)
            top_vals, top_idx = torch.topk(importance, k=top_k, dim=-1, sorted=False)
            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, D)
            return torch.gather(patch_tokens, dim=1, index=gather_idx), top_vals

    class ColPaliMaxSimScorer(nn.Module):
        def __init__(self, hidden_dim: int, proj_dim: int = 256):
            super().__init__()
            self.state_proj = MatryoshkaLinear(hidden_dim, proj_dim, bias=False)
            self.opt_proj = MatryoshkaLinear(hidden_dim, proj_dim, bias=False)
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, state_tokens: torch.Tensor, option_tokens: torch.Tensor, option_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            s_emb = F.normalize(self.state_proj(state_tokens), p=2, dim=-1)
            o_emb = F.normalize(self.opt_proj(option_tokens), p=2, dim=-1)
            sim_matrix = torch.einsum("bklp,bnp->bkln", o_emb, s_emb)
            max_sim, _ = sim_matrix.max(dim=-1)
            if option_mask is not None:
                max_sim = max_sim * option_mask
                scores = max_sim.sum(dim=-1) / option_mask.sum(dim=-1).clamp(min=1.0)
            else:
                scores = max_sim.mean(dim=-1)
            return self.scale * scores

    class PermutationEquivariantSetHead(nn.Module):
        def __init__(self, config: PEMLIConfig):
            super().__init__()
            self.config = config
            d = config.hidden_dim
            self.reasoning_slots = nn.Parameter(torch.randn(1, config.num_reasoning_slots, d) * 0.02)
            self.cross_attn = nn.MultiheadAttention(d, config.num_heads, batch_first=True)
            self.norm_slots = nn.LayerNorm(d)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=config.num_heads, dim_feedforward=d * 2,
                dropout=0.05, activation="gelu", batch_first=True, norm_first=True,
            )
            self.option_set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_set_layers)
            self.cond_gate = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d), nn.Sigmoid())
            self.decision_pointer = nn.Linear(d, 256, bias=False)
            self.candidate_pointer = nn.Linear(d, 256, bias=False)
            self.maxsim_scorer = ColPaliMaxSimScorer(d, proj_dim=256)

        def forward(self, state_tokens: torch.Tensor, question_vec: torch.Tensor, option_vecs: torch.Tensor, option_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
            B, K, d = option_vecs.shape
            slots = self.reasoning_slots[:, :, :d].expand(B, -1, -1)
            memory = torch.cat([question_vec.unsqueeze(1), state_tokens], dim=1)
            attended_slots, _ = self.cross_attn(query=slots, key=memory, value=memory)
            slots = self.norm_slots(slots + attended_slots)
            slot_summary = slots.mean(dim=1)
            modulation = self.cond_gate(torch.cat([slot_summary, question_vec], dim=-1)).unsqueeze(1)
            set_opts = self.option_set_encoder(option_vecs * modulation)
            q_ptr = self.decision_pointer(slot_summary + question_vec)
            c_ptr = self.candidate_pointer(set_opts)
            binding_logits = torch.bmm(c_ptr, q_ptr.unsqueeze(-1)).squeeze(-1) / math.sqrt(256.0)
            maxsim_logits = self.maxsim_scorer(state_tokens, option_tokens)
            total_logits = (1.0 - self.config.maxsim_weight) * binding_logits + self.config.maxsim_weight * maxsim_logits
            return {"logits": total_logits, "binding_logits": binding_logits, "maxsim_logits": maxsim_logits, "slot_summary": slot_summary}

    class HybridTeacherDistillationLoss(nn.Module):
        def __init__(self, config: PEMLIConfig):
            super().__init__()
            self.config = config

        def forward(self, student_logits: torch.Tensor, teacher_probs: torch.Tensor, gold_indices: torch.Tensor, state_q_emb: torch.Tensor, option_embs: torch.Tensor, mrl_slice_logits: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
            log_p = F.log_softmax(student_logits, dim=-1)
            loss_kl = F.kl_div(log_p, teacher_probs, reduction="batchmean")
            loss_ce = F.cross_entropy(student_logits, gold_indices)
            s_norm = F.normalize(state_q_emb, p=2, dim=-1).unsqueeze(1)
            o_norm = F.normalize(option_embs, p=2, dim=-1)
            cos_sims = (s_norm * o_norm).sum(dim=-1)
            pos_sim = cos_sims.gather(1, gold_indices.unsqueeze(1))
            mask = torch.ones_like(cos_sims, dtype=torch.bool)
            mask.scatter_(1, gold_indices.unsqueeze(1), False)
            hard_neg = cos_sims.masked_fill(~mask, -1e9).max(dim=-1, keepdim=True)[0]
            loss_margin = F.relu(self.config.contrastive_margin - (pos_sim - hard_neg)).mean()
            loss_mrl = torch.tensor(0.0, device=student_logits.device)
            if mrl_slice_logits:
                for sl in mrl_slice_logits:
                    loss_mrl = loss_mrl + F.kl_div(F.log_softmax(sl, dim=-1), teacher_probs, reduction="batchmean")
                loss_mrl = loss_mrl / len(mrl_slice_logits)
            total = (
                self.config.kl_teacher_weight * loss_kl
                + self.config.gold_ce_weight * loss_ce
                + self.config.contrastive_margin_weight * loss_margin
                + self.config.mrl_multislice_weight * loss_mrl
            )
            return {"loss": total, "loss_kl": loss_kl, "loss_ce": loss_ce, "loss_margin": loss_margin, "loss_mrl": loss_mrl}
