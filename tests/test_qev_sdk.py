"""Installed SDK entry points, assets and model bootstrap behavior."""

import hashlib
import json
from importlib.resources import files

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from qev import __version__, download, runtime
from qev.cli import main


def test_version_is_available_without_loading_model(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_star_is_a_link_unless_open_was_requested(monkeypatch, capsys):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    main(["star"])
    assert opened == []
    assert "https://github.com/ken-jo/qev" in capsys.readouterr().out
    main(["star", "--open"])
    assert opened == ["https://github.com/ken-jo/qev"]


def test_playground_assets_and_licenses_are_package_resources():
    resources = files("qev.playground")
    presets = json.loads(resources.joinpath("presets.json").read_text("utf-8"))
    assert set(presets) >= {"support", "photo", "photo_score", "photo_truth"}
    for number in range(1, 7):
        assert len(resources.joinpath(f"samples/sample-{number:02d}.jpg").read_bytes()) > 100
    assert "MIT License" in resources.joinpath("samples/LICENSE.txt").read_text("utf-8")


def test_complete_cache_is_reused_without_online_calls(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("QEV_OFFLINE", raising=False)
    monkeypatch.setattr(runtime, "download_checkpoint", lambda *a, **k: calls.append(("head", k)))
    monkeypatch.setattr(runtime, "download_backbone", lambda *a, **k: calls.append(("base", k)))
    folder, cache = runtime.prepare_checkpoint(tmp_path, cache_dir=tmp_path / "hub")
    assert folder == tmp_path.resolve()
    assert cache == str(tmp_path / "hub")
    assert calls == [("head", {"local_files_only": True}), ("base", {"local_files_only": True})]


def test_first_use_downloads_missing_weights(monkeypatch, tmp_path):
    calls, messages = [], []
    monkeypatch.delenv("QEV_OFFLINE", raising=False)

    def checkpoint(*args, local_files_only=False):
        calls.append(("head", local_files_only))
        if local_files_only:
            raise LocalEntryNotFoundError("Missing checkpoint")

    monkeypatch.setattr(runtime, "download_checkpoint", checkpoint)
    monkeypatch.setattr(runtime, "download_backbone", lambda *a, **k: calls.append(("base", k)))
    runtime.prepare_checkpoint(tmp_path, cache_dir=tmp_path / "hub", progress=messages.append)
    assert calls == [("head", True), ("head", False), ("base", {})]
    assert "4.6 GB" in messages[0]


def test_offline_missing_cache_has_actionable_error(monkeypatch, tmp_path):
    calls = []

    def checkpoint(*args, **kwargs):
        calls.append(kwargs)
        raise LocalEntryNotFoundError("Missing")

    monkeypatch.setattr(runtime, "download_checkpoint", checkpoint)
    with pytest.raises(RuntimeError, match="qev download"):
        runtime.prepare_checkpoint(tmp_path, offline=True)
    assert calls == [{"local_files_only": True}]


def test_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert runtime.resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        runtime.resolve_device("cuda")


def test_download_verifies_and_reuses_files(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint fixture")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(download, "CHECKPOINT_HASHES", {"head.safetensors": sha})
    calls = []

    def fetch(*args, **kwargs):
        calls.append(kwargs)
        return source

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    destination = tmp_path / "model"
    download.download_checkpoint(destination, str(tmp_path / "cache"))
    download.download_checkpoint(destination, str(tmp_path / "cache"), local_files_only=True)
    assert len(calls) == 1
    assert (destination / "head.safetensors").read_bytes() == source.read_bytes()
    assert list(destination.glob("*.partial")) == []
    (destination / "head.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Refusing to replace"):
        download.download_checkpoint(destination, str(tmp_path / "cache"))
    assert (destination / "head.safetensors").read_bytes() == b"changed"


def test_partial_backbone_cache_is_not_treated_as_complete(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(tmp_path))
    with pytest.raises(LocalEntryNotFoundError, match="Incomplete Qwen snapshot"):
        download.download_backbone(str(tmp_path), local_files_only=True)


def test_qev_home_does_not_depend_on_working_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("QEV_HOME", str(tmp_path / "models"))
    assert runtime.cache_home() == tmp_path / "models"


def test_playground_builds_from_packaged_assets():
    from qev.playground.app import build_demo

    class FakeEngine:
        device = "cpu"

        def predict(self, *args):
            raise AssertionError("Building the interface must not infer")

    demo = build_demo(FakeEngine())
    config = json.dumps(demo.get_config_file())
    assert "Sample photographs" in config
    assert "Star QEV on GitHub" in config
    assert "Score visibility" in config
    demo.close()
