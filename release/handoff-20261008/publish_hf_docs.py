"""Publish only the merged model card and its checksum; never load model payloads.

Default mode is read-only preflight. --publish uses one parent-guarded Hub commit.
The receipt is stdout; the caller saves it with the native file editing tool.
"""

import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "ken-jo/qev"
SOURCE_COMMIT = "5e5cf9b310ead30d3fe6b72d4d59508e5c2545b3"
PARENT = "1b801c0a596d931a2fa2f587bf5e5eb70f096fea"
BASELINE = {
    "README.md": "97ca455a2014bde4aefe8380cb65a9f690ad09f2dbd91c25580f988ab443037f",
    "checksums.json": "7ad3cc9239bb92e5e743ae225b8baff0c35718fe17e9625dfe7d20d05f699d61",
    "publication.json": "b1452a1e57cbc6cdfcc6a7095552cb6c6f25b734eb1e96e442c260218bca60c9",
}
CANDIDATE = "af65ab9704867e93b10f06813d69e2c7ce94fc9f9e7e27a60b0f2f81cac46323"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def anonymous_get(url):
    # Normal public HTTP reads; no authentication headers or payload caches.
    request = Request(url, headers={"User-Agent": "QEV-docs-verification"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def remote_file(revision, name):
    return anonymous_get(f"https://huggingface.co/{REPO}/resolve/{revision}/{name}")


def tree(api, revision):
    result = {}
    for item in api.list_repo_tree(REPO, repo_type="model", revision=revision, recursive=True):
        if not hasattr(item, "blob_id"):
            continue
        lfs = item.lfs
        result[item.path] = {
            "blobId": item.blob_id,
            "size": item.size,
            "lfsSha256": lfs.sha256 if lfs else None,
            "lfsSize": lfs.size if lfs else None,
        }
    return result


def tree_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    anonymous = HfApi(token=False)
    assert anonymous.model_info(REPO, revision="main").sha == PARENT, (
        "HF parent changed; review required"
    )
    readme = (Path(__file__).resolve().parents[2] / "MODEL_CARD.md").read_bytes()
    assert digest(readme) == CANDIDATE, "Candidate card changed; review required"
    github = json.loads(anonymous_get(
        f"https://api.github.com/repos/{REPO}/contents/MODEL_CARD.md?ref={SOURCE_COMMIT}"
    ))
    assert base64.b64decode(github["content"]) == readme, "Local card differs from merged source"
    before_bytes = {name: remote_file(PARENT, name) for name in BASELINE}
    assert {name: digest(value) for name, value in before_bytes.items()} == BASELINE
    before_checksums = json.loads(before_bytes["checksums.json"])
    assert len(before_checksums) == 208
    old_marker = f'"README.md": "{BASELINE["README.md"]}"'.encode()
    new_marker = f'"README.md": "{CANDIDATE}"'.encode()
    assert before_bytes["checksums.json"].count(old_marker) == 1
    checksums = before_bytes["checksums.json"].replace(old_marker, new_marker, 1)
    after_checksums = json.loads(checksums)
    assert before_checksums.keys() == after_checksums.keys()
    changed_checksums = {
        key for key in before_checksums if before_checksums[key] != after_checksums[key]
    }
    assert changed_checksums == {"README.md"}
    before_tree = tree(anonymous, PARENT)
    assert len(before_tree) == 210
    receipt = {
        "scope": "documentation-only",
        "source_commit": SOURCE_COMMIT,
        "source_card_sha256": CANDIDATE,
        "parent_hf_commit": PARENT,
        "repo_id": REPO,
        "planned_changed_paths": ["README.md", "checksums.json"],
        "frontmatter_changes": [
            "pipeline_tag: image-text-to-text", "tag: classification", "tag: structured-output"
        ],
        "before_tree_sha256": tree_digest(before_tree),
        "file_count": len(before_tree),
        "checksum_entries": len(before_checksums),
        "before_files": {
            name: {"sha256": digest(value), "size": len(value)}
            for name, value in before_bytes.items()
        },
        "candidate_files": {
            "README.md": {"sha256": digest(readme), "size": len(readme)},
            "checksums.json": {"sha256": digest(checksums), "size": len(checksums)},
        },
        "inherited_publication": {
            "hf_commit": PARENT,
            "path": "publication.json",
            "sha256": BASELINE["publication.json"],
            "package_checksums_sha256": BASELINE["checksums.json"],
            "local_receipts": [
                "release/model-inference.json", "release/verification.json",
                "release/huggingface-publication.json",
            ],
        },
        "preserved_model_identity": {
            name: before_checksums[name]
            for name in ["head.safetensors", "manifest.json", "runtime/qev-0.2.1-py3-none-any.whl"]
        },
        "model_execution_performed": False,
        "sdk_published": False,
        "dataset_changed": False,
    }
    if not args.publish:
        receipt.update({"status": "preflight-passed", "mutation_performed": False})
        print(json.dumps(receipt))
        return
    commit = HfApi().create_commit(
        repo_id=REPO,
        repo_type="model",
        revision="main",
        parent_commit=PARENT,
        operations=[
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme),
            CommitOperationAdd(path_in_repo="checksums.json", path_or_fileobj=checksums),
        ],
        commit_message="Clarify QEV model card and unreleased SDK compatibility",
        commit_description="Documentation-only update; model and runtime artifacts unchanged.",
    )
    after_tree = tree(anonymous, commit.oid)
    assert before_tree.keys() == after_tree.keys()
    changed = sorted(name for name in before_tree if before_tree[name] != after_tree[name])
    assert changed == ["README.md", "checksums.json"]
    assert remote_file(commit.oid, "README.md") == readme
    assert remote_file(commit.oid, "checksums.json") == checksums
    assert anonymous.model_info(REPO, revision="main").sha == commit.oid
    receipt.update({
        "status": "published-and-verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_commit": commit.oid,
        "commit_url": str(commit.commit_url),
        "changed_paths": changed,
        "after_tree_sha256": tree_digest(after_tree),
        "unchanged_other_files": 208,
        "unchanged_other_checksum_entries": 207,
        "anonymous_downloads_match": True,
        "live_main_matches": True,
        "mutation_performed": True,
    })
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
