"""Publish already-audited payloads to the authenticated owner's Hugging Face account."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "release/verification.json").read_text("utf-8"))
    if not audit.get("passed"):
        raise ValueError("Publication audit has not passed")
    api = HfApi()
    identity = api.whoami()
    if identity["name"] != args.owner:
        raise ValueError("Authenticated account differs from the explicit owner")
    published = {}
    for kind, name, folder in (
        ("model", "vision-qev", args.model),
        ("dataset", "vision-qev-data", args.data),
    ):
        check_key = "model_checksums_sha256" if kind == "model" else "data_checksums_sha256"
        if sha(folder / "checksums.json") != audit[check_key]:
            raise ValueError("Payload changed since the publication audit")
        checks = json.loads((folder / "checksums.json").read_text("utf-8"))
        for path, expected in checks.items():
            target = (folder / path).resolve()
            if not target.is_relative_to(folder.resolve()) or sha(target) != expected:
                raise ValueError("Payload checksum mismatch")
        repo = args.owner + "/" + name
        try:
            info = api.repo_info(repo, repo_type=kind)
            existing = api.list_repo_files(repo, repo_type=kind)
            allowed = set(checks) | {"checksums.json", ".gitattributes"}
            if set(existing) - allowed:
                raise ValueError("Existing repository has unrelated files; refusing overwrite")
            if "checksums.json" in existing:
                previous = Path(hf_hub_download(repo, "checksums.json", repo_type=kind))
                if json.loads(previous.read_text("utf-8")) != checks:
                    raise ValueError("Existing repository contains a different release")
            if info.private:
                raise ValueError("Existing repo is private; review visibility before publication")
        except RepositoryNotFoundError:
            api.create_repo(repo, repo_type=kind, private=False)
        commit = api.upload_folder(
            repo_id=repo,
            repo_type=kind,
            folder_path=folder,
            allow_patterns=[*checks, "checksums.json"],
            commit_message="Release Vision QEV 0.1.0: Qwen3.5-2B typed decisions",
            commit_description=(
                "Audited English release with exact runtime/weight provenance, "
                "source-specific data licenses and published limitations."
            ),
        )
        revision = commit.oid
        remote = {
            entry.path: entry
            for entry in api.list_repo_tree(repo, repo_type=kind, revision=revision, recursive=True)
            if hasattr(entry, "blob_id")
        }
        for name_in_repo in [*checks, "checksums.json"]:
            raw = (folder / name_in_repo).read_bytes()
            entry = remote[name_in_repo]
            if entry.lfs:
                remote_sha = (
                    entry.lfs.get("sha256") if isinstance(entry.lfs, dict) else entry.lfs.sha256
                )
                if remote_sha != hashlib.sha256(raw).hexdigest():
                    raise ValueError("Remote LFS hash mismatch: " + name_in_repo)
            else:
                blob = b"blob " + str(len(raw)).encode() + b"\0" + raw
                if entry.blob_id != hashlib.sha1(blob).hexdigest():
                    raise ValueError("Remote Git blob mismatch: " + name_in_repo)
        anonymous = Path(
            hf_hub_download(
                repo,
                "checksums.json",
                repo_type=kind,
                revision=revision,
                token=False,
                cache_dir=root / ".cache/public-download-verification",
            )
        )
        if anonymous.read_bytes() != (folder / "checksums.json").read_bytes():
            raise ValueError("Anonymous public download verification failed")
        url = "https://huggingface.co/" + ("datasets/" if kind == "dataset" else "") + repo
        published[kind] = {
            "url": url,
            "revision": revision,
            "public": True,
            "all_remote_file_hashes_match": True,
            "anonymous_checksums_download_passed": True,
            "verified_files": len(checks) + 1,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"published": kind, **published[kind]}), flush=True)


if __name__ == "__main__":
    main()
