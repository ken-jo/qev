"""Decision Index vision engine for released QEV checkpoints (ken-jo/QEV-0.8B, ken-jo/QEV-2B).

Loads the released files with the qev loader and answers each row in one forward pass, no generation.
Images go to the model's own processor at native resolution, capped at 1,638,400 px per image by the
checkpoint's ``accurate`` mode; there is no lower cap and no shared multi-image budget.

    decision-index vision pipeline --dir work/vision \
        --engine qev_decision_index:QEVEngine --option model=ken-jo/QEV-2B --out runs/qev-2b

Declared limits: at most 4 images per request and 8,192 input tokens (longer requests are reported as
unsupported, never truncated).
"""
from pathlib import Path

import torch
from decision_index.engines.base import Engine, Unsupported
from decision_index.vision.engines import single_choice


class QEVEngine(Engine):
    name = "qev"

    def __init__(self, model="ken-jo/QEV-2B", revision=None, mode="accurate", device="cuda", image_root=None, **options):
        super().__init__(**options)
        self.root = Path(image_root) if image_root else None
        from veyra.option_model import OptionModel
        path = Path(model)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(model, revision=revision))
        self.m = OptionModel.load(path, device=device, mode=mode)
        self.m.eval()
        self.model = model
        self.provenance = {"kind": "qev-option-readout", "model": model, "revision": revision, "mode": mode,
                           "policy": "single forward pass, typed option readout, no generation; native resolution "
                                     "up to 1,638,400 px per image; readout prefix ignored"}

    def warmup(self):
        pass

    def __call__(self, state=None, questions=None, images=(), prompt=None):
        from veyra.schema import DecisionRequest
        key, q, keys = single_choice(questions)
        if not images:
            raise Unsupported("no image")
        if len(images) > 4:
            raise Unsupported("QEV request contract takes at most 4 images")
        images = [str(self.root / p) if self.root and not Path(p).is_absolute() else str(p) for p in images]
        crit = {}
        for k, v in q["criteria"].items():  # option texts must be distinct for the option readout
            v = str(v).strip() or k
            while v in crit.values():
                v = f"{v} ({k})"
            crit[k] = v[:8000]
        instr = q["instructions"].strip()[:8000] or "Choose the correct option."
        req = DecisionRequest.model_validate({"state": {"text": "", "images": [{"path": p} for p in images]},
                                              "questions": {key: {"type": "choice", "instructions": instr, "criteria": crit}}})
        try:
            with torch.inference_mode():
                out = self.m.predict(req)
        except ValueError as e:  # over the token budget: refused, not truncated
            raise Unsupported(str(e)) from e
        a = out["answers"][key]
        probs = {k: float(a["probabilities"][k]) for k in keys}
        s = sum(probs.values())
        probs = {k: v / s for k, v in probs.items()}
        pixels, tokens = self._image_budget(images)
        return ({"model": self.model, "answers": {key: {"type": "choice", "choice": a["choice"], "probabilities": probs}},
                 "usage": out.get("usage")},
                {"input_tokens": out["usage"]["input_tokens"], "image_pixels": pixels, "image_tokens": tokens})

    def _image_budget(self, images):
        """Pixels and visual tokens per image after the processor's resize, for the run's effective_image_budget."""
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        ip = self.m.encoder.processor.image_processor
        factor = ip.patch_size * ip.merge_size
        pixels, tokens = [], 0
        for p in images:
            with Image.open(p) as im:
                h, w = smart_resize(im.height, im.width, factor, ip.size["shortest_edge"], ip.size["longest_edge"])
            pixels.append(h * w)
            tokens += (h // factor) * (w // factor)
        return pixels, tokens
