"""Real frozen-backbone baseline. No synthetic fallback or generated answer tokens.

The embedding artifact is a learned task projection over pinned HF token features;
the decision artifact is a candidate-conditioned cross-attention head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from fastapi import Request
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from veyra.candidates import candidates_for
from veyra.image_inputs import image_paths, open_images
from veyra.probability import Calibration, typed_answer
from veyra.schema import DecisionRequest


@dataclass
class RealConfig:
    backbone_id: str = "google/embeddinggemma-2"
    revision: str = ""  # Required immutable upstream commit.
    token_dim: int = 768
    width: int = 256
    heads: int = 4
    cache_entries: int = 16
    max_length: int = 4096


class TokenBackbone:
    def __init__(self, config: RealConfig, device: str):
        from transformers import AutoConfig, AutoModel, AutoProcessor

        if not re.fullmatch(r"[0-9a-f]{40}", config.revision):
            raise ValueError("backbone revision must be a pinned 40-character commit SHA")
        self.config, self.device = config, device
        hf_config = AutoConfig.from_pretrained(config.backbone_id, revision=config.revision)
        if hf_config.model_type != "embedding_gemma2":
            raise ValueError("This baseline explicitly supports EmbeddingGemma 2 only")
        hf_config.audio_config = None
        dtype = (
            torch.bfloat16
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.model = (
            AutoModel.from_pretrained(
                config.backbone_id, revision=config.revision, config=hf_config, torch_dtype=dtype
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.processor = AutoProcessor.from_pretrained(config.backbone_id, revision=config.revision)

    @torch.no_grad()
    def encode(self, text: str, images=None):
        kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
        if images:
            kwargs["images"] = [images]
        inputs = self.processor(**kwargs)
        if inputs["input_ids"].shape[1] > self.config.max_length:
            raise ValueError("input exceeds token budget; evidence is never silently truncated")
        inputs = inputs.to(self.device)
        output = self.model(**inputs).last_hidden_state
        if output.shape[-1] != self.config.token_dim:
            raise ValueError(f"unexpected token dimension: {output.shape[-1]}")
        tokens = output[0, inputs["attention_mask"][0].bool()].float()
        if not torch.isfinite(tokens).all():
            raise ValueError("non-finite backbone features")
        return tokens.detach().cpu()


class EmbeddingAdapter(nn.Module):
    def __init__(self, config: RealConfig):
        super().__init__()
        self.project = nn.Linear(config.token_dim, config.width, bias=False)
        self.norm = nn.LayerNorm(config.width)

    def forward(self, tokens):
        return F.normalize(self.norm(self.project(tokens)), dim=-1)


class DecisionHead(nn.Module):
    """Same verifier per candidate: candidate IDs/order never enter the network."""

    def __init__(self, config: RealConfig):
        super().__init__()
        d = config.width
        self.query = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, config.heads, batch_first=True, dropout=0)
        self.score = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.sim_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, state, question, options):
        q = question.mean(0).expand(len(options), -1)
        o = torch.stack([x.mean(0) for x in options])
        queries = self.query(torch.cat([q, o], -1))
        # A single batch with K independent queries over shared evidence.
        evidence, _ = self.attn(queries[None], state[None], state[None], need_weights=False)
        evidence = evidence[0]
        logits = self.score(torch.cat([q, o, evidence, o * evidence], -1)).squeeze(-1)
        alignment = torch.stack([(x @ state.T).max(-1).values.mean() for x in options])
        return logits + self.sim_scale * alignment


class RealDecisionModel(nn.Module):
    def __init__(self, config: RealConfig, backbone=None, device="cpu"):
        super().__init__()
        self.config = config
        self.embedding = EmbeddingAdapter(config)
        self.head = DecisionHead(config)
        # Backbone intentionally outside nn.Module: frozen and referenced by pinned SHA.
        self.backbone = backbone
        self.calibration = Calibration(always_abstain_types=("choice", "score", "noul"))
        self.trained = False
        self.cache = OrderedDict()
        self._lock = threading.RLock()
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def raw_features(self, request, root):
        if self.backbone is None:
            raise RuntimeError("real backbone is required; no mock inference fallback")
        paths = image_paths(request.state.images, root)
        digest = hashlib.sha256(request.state.text.encode())
        for path in paths:
            data = path.read_bytes()
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
        key = digest.hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            state = self.cache[key]
        else:
            image_text = " ".join(f"Image {i + 1}: <|image|>" for i in range(len(paths)))
            with open_images(request.state.images, root) as images:
                state = self.backbone.encode(
                    "title: none | text: " + request.state.text + " " + image_text, images
                )
            if self.config.cache_entries:
                self.cache[key] = state
                while len(self.cache) > self.config.cache_entries:
                    self.cache.popitem(last=False)
        features = {}
        for name, question in request.questions.items():
            q = self.backbone.encode("task: question answering | query: " + question.instructions)
            options = [
                self.backbone.encode("title: none | text: " + c.description)
                for c in candidates_for(question)
            ]
            features[name] = (state, q, options)
        return features

    def logits_from_features(self, features):
        state, question, options = features

        def emb(x):
            return self.embedding(x.to(self.device))

        return self.head(emb(state), emb(question), [emb(x) for x in options])

    @torch.no_grad()
    def predict(self, request, image_root=None):
        if not self.trained:
            raise RuntimeError("trained checkpoint required")
        with self._lock:
            self.eval()
            if isinstance(request, dict):
                request = DecisionRequest.from_systemone_json(json.dumps(request))
            features = self.raw_features(request, image_root)
            return {
                "model": "qev-pemli-real",
                "architecture": "candidate_evidence_verifier",
                "backbone": {"model_id": self.config.backbone_id, "revision": self.config.revision},
                "answers": {
                    name: typed_answer(
                        q, self.logits_from_features(features[name]), self.calibration
                    )
                    for name, q in request.questions.items()
                },
                "usage": {
                    "input_tokens": sum(
                        len(x) for f in features.values() for x in [f[0], f[1], *f[2]]
                    ),
                    "output_tokens": 0,
                    "input_images": len(request.state.images),
                },
            }

    def save(self, directory, provenance):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        for name, module in [("embedding", self.embedding), ("decision", self.head)]:
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()},
                str(directory / f"{name}.safetensors"),
            )
        manifest = {
            "format": "qev-pemli-real-v1",
            "config": asdict(self.config),
            "trained": self.trained,
            "calibration": self.calibration.to_dict(),
            "embedding_scope": "learned token projection; upstream backbone frozen",
            "provenance": provenance,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory, device="cpu", with_backbone=True):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format"] != "qev-pemli-real-v1" or not manifest["trained"]:
            raise ValueError("unsupported or untrained checkpoint")
        config = RealConfig(**manifest["config"])
        model = cls(config, TokenBackbone(config, device) if with_backbone else None, device)
        model.embedding.load_state_dict(load_file(str(directory / "embedding.safetensors")))
        model.head.load_state_dict(load_file(str(directory / "decision.safetensors")))
        model.calibration = Calibration.from_dict(manifest["calibration"])
        model.trained = True
        return model.eval()


def train(records_path, output, config, device, epochs=2, lr=1e-3, seed=42):
    from veyra.data import read_records

    if epochs < 1 or Path(output).exists():
        raise ValueError("positive epochs and a new checkpoint output are required")
    torch.manual_seed(seed)
    records_path = Path(records_path)
    records = read_records(records_path)
    selected = [r for r in records if r.split == "train"]
    if not selected:
        raise ValueError("no training records")
    model = RealDecisionModel(config, TokenBackbone(config, device), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    log = []
    # Recompute frozen features per record; bounded RAM and no retained autograd graph.
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        for idx in torch.randperm(len(selected), generator=generator).tolist():
            record = selected[idx]
            features = model.raw_features(record.request, records_path.parent)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for name, question in record.request.questions.items():
                logits = model.logits_from_features(features[name])
                target = torch.tensor(
                    [record.targets[name][c.key] for c in candidates_for(question)], device=device
                )
                losses.append(-(target * F.log_softmax(logits, -1)).sum())
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            row = {"epoch": epoch + 1, "record_id": record.id, "loss": float(loss.detach())}
            log.append(row)
            print(json.dumps(row), flush=True)
    model.trained = True
    model.save(
        output,
        {
            "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
            "training_groups": sorted({r.group_id for r in selected}),
            "epochs": epochs,
            "seed": seed,
            "lr": lr,
            "objective": "soft-target cross entropy",
            "teacher": None,
        },
    )
    (Path(output) / "training.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in log), encoding="utf-8"
    )
    return model


@torch.no_grad()
def calibrate(checkpoint, records_path, output, device):
    """Empirical group-split policy fitting; no conformal or OOD guarantee."""
    from veyra.constants import QUESTION_TYPES
    from veyra.data import read_records

    model = RealDecisionModel.load(checkpoint, device)
    records_path = Path(records_path)
    records = read_records(records_path)
    manifest = json.loads((Path(checkpoint) / "manifest.json").read_text())
    selected = [r for r in records if r.split == "calibration"]
    if not selected:
        raise ValueError("separate calibration split required")
    if {r.group_id for r in selected} & set(manifest["provenance"]["training_groups"]):
        raise ValueError("calibration groups overlap training")
    rows = []
    for r in selected:
        features = model.raw_features(r.request, records_path.parent)
        bucket = int(hashlib.sha256(r.group_id.encode()).hexdigest()[:8], 16) % 2
        for name, q in r.request.questions.items():
            target = torch.tensor([r.targets[name][c.key] for c in candidates_for(q)])
            rows.append((q.type, bucket, model.logits_from_features(features[name]).cpu(), target))
    temps, thresholds, fitted, abstention_fitted, always = [], [], [], [], []
    report = {}
    for kind in QUESTION_TYPES:
        temp_rows = [r for r in rows if r[0] == kind and r[1] == 0]
        policy_rows = [r for r in rows if r[0] == kind and r[1] == 1]
        t = 1.0
        threshold = 1.0
        accepted = 0
        if temp_rows:
            grid = torch.logspace(0, 1, 81)
            losses = [
                sum(float(-(r[3] * F.log_softmax(r[2] / value, -1)).sum()) for r in temp_rows)
                for value in grid
            ]
            t = float(grid[min(range(len(grid)), key=lambda i: losses[i])])
            fitted.append(kind)
        if temp_rows and policy_rows:
            pairs = []
            for r in policy_rows:
                p = (r[2] / t).softmax(-1)
                pairs.append((float(p.max()), 1 - float(r[3][int(p.argmax())])))
            for tau in sorted({x[0] for x in pairs}):
                errors = [e for c, e in pairs if c >= tau]
                if len(errors) >= 20 and sum(errors) / len(errors) <= 0.15:
                    threshold, accepted = tau, len(errors)
                    abstention_fitted.append(kind)
                    break
        if not accepted:
            always.append(kind)
        temps.append(t)
        thresholds.append(threshold)
        report[kind] = {
            "temperature_questions": len(temp_rows),
            "policy_questions": len(policy_rows),
            "accepted": accepted,
        }
    model.calibration = Calibration(
        tuple(temps), tuple(thresholds), tuple(fitted), tuple(abstention_fitted), tuple(always)
    )
    provenance = dict(manifest["provenance"])
    provenance["calibration"] = {
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "groups": sorted({r.group_id for r in selected}),
        "method": "temperature fit and empirical policy on disjoint groups; no risk guarantee",
        "types": report,
    }
    model.save(output, provenance)
    return report


def create_app(model, image_root):
    from fastapi import FastAPI, HTTPException

    app = FastAPI()

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            payload = DecisionRequest.from_systemone_json((await request.body()).decode())
            return model.predict(payload, image_root)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--records", type=Path, required=True)
    tr.add_argument("--output", type=Path, required=True)
    tr.add_argument("--revision", required=True)
    tr.add_argument("--epochs", type=int, default=2)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--records", type=Path, required=True)
    cal.add_argument("--output", type=Path, required=True)
    for command in [tr, cal, sub.add_parser("predict"), sub.add_parser("serve")]:
        command.add_argument("--device", default="cuda")
        if command is not tr:
            command.add_argument("--checkpoint", type=Path, required=True)
            if command is not cal:
                command.add_argument("--image-root", type=Path, required=True)
    pred = sub.choices["predict"]
    pred.add_argument("--request", type=Path, required=True)
    server = sub.choices["serve"]
    server.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1:
            parser.error("epochs must be positive")
        train(
            args.records, args.output, RealConfig(revision=args.revision), args.device, args.epochs
        )
    elif args.command == "calibrate":
        print(json.dumps(calibrate(args.checkpoint, args.records, args.output, args.device)))
    else:
        model = RealDecisionModel.load(args.checkpoint, args.device)
        if args.command == "predict":
            req = DecisionRequest.from_systemone_json(args.request.read_text(encoding="utf-8"))
            print(
                json.dumps(model.predict(req, args.image_root), ensure_ascii=False, allow_nan=False)
            )
        else:
            import uvicorn

            uvicorn.run(create_app(model, args.image_root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
