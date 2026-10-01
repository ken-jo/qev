import json

import pytest
import torch

from veyra.backbone import EncoderConfig
from veyra.checkpoint import load_head, save_checkpoint
from veyra.head import DecisionHead, HeadConfig
from veyra.probability import Calibration


def test_round_trip_and_guard(tmp_path):
    torch.manual_seed(4)
    head = DecisionHead(HeadConfig(12, 16, 1, 4))
    path = tmp_path / "trained"
    save_checkpoint(
        path, head, EncoderConfig(), Calibration(), {"completed": True, "optimizer_steps": 3}
    )
    restored, config, calibration, _ = load_head(path)
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    assert config.encoding == "packed-v1"
    assert not calibration.fitted_types
    with pytest.raises(FileExistsError):
        save_checkpoint(path, head, config, calibration, {})
    fresh = tmp_path / "untrained"
    save_checkpoint(fresh, head, config, calibration, {"completed": False, "optimizer_steps": 0})
    with pytest.raises(ValueError, match="completed head training"):
        load_head(fresh)


def test_checksum_and_revision_enforced(tmp_path):
    head = DecisionHead(HeadConfig(12, 16, 1, 4))
    save_checkpoint(
        tmp_path, head, EncoderConfig(), Calibration(), {"completed": True, "optimizer_steps": 1}
    )
    manifest_path = tmp_path / "manifest.json"
    original = json.loads(manifest_path.read_text())
    changed = {**original, "backbone": {"model_id": "other", "revision": "main"}}
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="backbone"):
        load_head(tmp_path)
    manifest_path.write_text(json.dumps(original))
    with (tmp_path / "head.safetensors").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_head(tmp_path)
