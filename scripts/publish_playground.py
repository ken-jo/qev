"""Publish the verified English playground on explicitly free Spaces hardware."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", choices=["cpu-basic", "zero-a10g"], default="zero-a10g")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "apps/hf_space"
    allowed = [
        "app.py",
        "engine.py",
        "presets.json",
        "requirements.txt",
        "README.md",
        "LICENSE",
        "NOTICE",
        "samples/LICENSE.txt",
        "samples/README.md",
        "samples/provenance.json",
    ] + [f"samples/sample-{index:02d}.jpg" for index in range(1, 7)]
    provenance = json.loads((folder / "samples/provenance.json").read_text("utf-8"))
    for image in provenance["images"]:
        actual = hashlib.sha256((folder / "samples" / image["file"]).read_bytes()).hexdigest()
        if actual != image["sha256"]:
            raise ValueError("Sample photograph checksum mismatch")
    for relative in allowed:
        path = folder / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix != ".jpg" and re.search("[\uac00-\ud7a3]", path.read_text("utf-8")):
            raise ValueError("The public playground must be English")
    verification = json.loads((root / "runs/verification/space-http.json").read_text("utf-8"))
    if not verification.get("passed") or verification.get("real_model_requests") != 4:
        raise ValueError("Verify text and all three image decision types before publication")
    for name in ("app.py", "engine.py", "presets.json", "requirements.txt"):
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if verification.get("source_hashes", {}).get(name) != actual:
            raise ValueError("Playground source changed after HTTP verification")
    api = HfApi()
    if api.whoami()["name"] != "ken-jo":
        raise ValueError("Unexpected publishing account")
    repo = "ken-jo/qwen3.5-classification"
    try:
        info = api.repo_info(repo, repo_type="space")
        if info.private or info.sdk != "gradio":
            raise ValueError("Existing Space has incompatible visibility or SDK")
        files = set(api.list_repo_files(repo, repo_type="space"))
        if files - set(allowed) - {".gitattributes"}:
            raise ValueError("Existing Space contains unrelated files")
    except RepositoryNotFoundError:
        api.create_repo(
            repo,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            space_hardware=args.hardware,
            space_variables=[{"key": "QEV_ZERO_GPU", "value": "1"}]
            if args.hardware == "zero-a10g"
            else None,
        )
        info = api.repo_info(repo, repo_type="space")
    runtime = api.get_space_runtime(repo)
    selected = runtime.requested_hardware or runtime.hardware
    if selected and getattr(selected, "value", selected) != args.hardware:
        raise ValueError("Space hardware differs from the explicit free-hardware request")
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="space",
        folder_path=folder,
        allow_patterns=allowed,
        parent_commit=info.sha,
        commit_message="Add the verified English text and image playground",
    )
    print(
        json.dumps(
            {
                "url": f"https://huggingface.co/spaces/{repo}",
                "revision": commit.oid,
                "hardware_requested": args.hardware,
                "deployment_verification_pending": True,
            }
        )
    )


if __name__ == "__main__":
    main()
