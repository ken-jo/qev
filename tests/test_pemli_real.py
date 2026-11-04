"""Offline neural contract tests; these do not establish real-backbone accuracy."""

import json

import torch
from PIL import Image

from qev.pemli_real import RealConfig, RealDecisionModel, create_app
from veyra.schema import DecisionRequest


def config():
    return RealConfig(token_dim=16, width=16, heads=4)


def features():
    torch.manual_seed(1)
    return torch.randn(12, 16), torch.randn(3, 16), [torch.randn(4, 16) for _ in range(3)]


def test_gradient_update_and_checkpoint_roundtrip(tmp_path):
    model = RealDecisionModel(config())
    f = features()
    before = model.logits_from_features(f).detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(4):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(
            model.logits_from_features(f)[None], torch.tensor([2])
        ).backward()
        assert model.embedding.project.weight.grad.abs().sum() > 0
        assert model.head.attn.in_proj_weight.grad.abs().sum() > 0
        opt.step()
    after = model.logits_from_features(f).detach()
    assert not torch.allclose(before, after)
    model.trained = True
    model.save(tmp_path / "model", {"training_groups": ["a"]})
    loaded = RealDecisionModel.load(tmp_path / "model", with_backbone=False)
    torch.testing.assert_close(after, loaded.logits_from_features(f))


def test_permutation_equivariance():
    model = RealDecisionModel(config()).eval()
    s, q, o = features()
    expected = model.logits_from_features((s, q, o))
    order = [2, 0, 1]
    actual = model.logits_from_features((s, q, [o[i] for i in order]))
    torch.testing.assert_close(actual, expected[order])


def test_evidence_and_question_affect_logits():
    model = RealDecisionModel(config()).eval()
    s, q, o = features()
    expected = model.logits_from_features((s, q, o))
    assert not torch.allclose(expected, model.logits_from_features((-s, q, o)))
    assert not torch.allclose(expected, model.logits_from_features((s, -q, o)))


class PixelEncoder:
    """Test-only pixel-dependent encoder; production never instantiates this."""

    def encode(self, text, images=None):
        if images:
            return torch.full((3, 16), float(images[0].getpixel((0, 0))[0]))
        return torch.arange(48, dtype=torch.float32).reshape(3, 16)


def test_image_bytes_not_path_controls_cache(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(image)
    req = DecisionRequest.from_systemone_json(
        json.dumps(
            {
                "state": {"images": [{"path": "image.png"}]},
                "questions": {"q": {"type": "choice", "criteria": {"a": "red", "b": "blue"}}},
            }
        )
    )
    model = RealDecisionModel(config(), PixelEncoder())
    before = model.raw_features(req, tmp_path)["q"][0]
    Image.new("RGB", (4, 4), (255, 255, 255)).save(image)
    after = model.raw_features(req, tmp_path)["q"][0]
    assert not torch.equal(before, after)


def test_json_three_types_and_http(tmp_path):
    from fastapi.testclient import TestClient

    model = RealDecisionModel(config(), PixelEncoder())
    model.trained = True
    payload = {
        "state": "payment failed",
        "questions": {
            "c": {"type": "choice", "criteria": {"a": "refund", "b": "repair"}},
            "s": {"type": "score", "criteria": ["low", "high"]},
            "n": {"type": "noul", "instructions": "Payment failed?"},
        },
    }
    result = model.predict(payload, tmp_path)
    assert set(result["answers"]) == {"c", "s", "n"}
    assert isinstance(result["answers"]["c"]["choice"], str)
    assert isinstance(result["answers"]["s"]["score"], float)
    assert 0 <= result["answers"]["n"]["noul"] <= 1
    assert all(a["abstained"] and not a["calibrated"] for a in result["answers"].values())
    assert result["usage"]["output_tokens"] == 0
    json.dumps(result, allow_nan=False)
    client = TestClient(create_app(model, tmp_path))
    assert client.post("/v1/systemone", json=payload).status_code == 200
    assert client.post("/v1/systemone", json={}).status_code == 422


def test_training_driver_writes_learned_artifacts(tmp_path, monkeypatch):
    import qev.pemli_real as module
    from veyra.data import TrainingRecord

    req = DecisionRequest.from_systemone_json(
        json.dumps(
            {
                "state": "payment failed",
                "questions": {"q": {"type": "choice", "criteria": {"a": "refund", "b": "repair"}}},
            }
        )
    )
    records = [
        TrainingRecord.model_validate(
            {
                "id": f"r{i}",
                "group_id": f"g{i}",
                "split": split,
                "family": "contract",
                "language": "en",
                "request": req,
                "targets": {"q": {"a": 1.0, "b": 0.0}},
                "source": {
                    "id": "test-only",
                    "revision": "1",
                    "license": "Apache-2.0",
                    "url": "https://example.org",
                },
            }
        )
        for i, split in enumerate(["train", "train", "calibration", "test"])
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("".join(r.model_dump_json() + "\n" for r in records))
    monkeypatch.setattr(module, "TokenBackbone", lambda cfg, device: PixelEncoder())
    trained = module.train(path, tmp_path / "trained", config(), "cpu", epochs=1)
    assert trained.trained
    assert len((tmp_path / "trained" / "training.jsonl").read_text().splitlines()) == 2
    reloaded = module.RealDecisionModel.load(tmp_path / "trained", with_backbone=False)
    torch.testing.assert_close(trained.embedding.project.weight, reloaded.embedding.project.weight)
    manifest = json.loads((tmp_path / "trained" / "manifest.json").read_text())
    assert manifest["provenance"]["training_groups"] == ["g0", "g1"]
