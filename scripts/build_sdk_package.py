"""Build the Hugging Face SDK package from the previous package and a freshly built wheel.

Weights, manifest, licenses and the release comparison are copied byte-for-byte from the
previous package; the card, docs, example, reproducibility files and the new runtime wheel
come from this checkout. The historical evidence folder, earlier wheels and every other file
carrying the legacy runtime name are dropped; they stay at ARCHIVE_REVISION. The pinned
manifest is the one exception, and in the new wheel only the trained result marker may
carry the legacy spelling.
"""

import argparse
import hashlib
import json
import posixpath
import re
import shutil
import tomllib
import zipfile
from pathlib import Path

from qev.download import CHECKPOINT_HASHES

FROM_CHECKOUT = {
    "README.md": "MODEL_CARD.md",
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
    "ROADMAP.md": "ROADMAP.md",
    "sample_request.json": "examples/request.json",
    "reproducibility/pyproject.toml": "pyproject.toml",
    "reproducibility/uv.lock": "uv.lock",
    **{
        f"docs/{name}": f"docs/{name}"
        for name in (
            "LAYA_COMPARISON.md",
            "MODEL_SIZE.md",
            "COMPACT_MODEL.md",
            "PLAYGROUND.md",
            "API.md",
        )
    },
    **{
        f"examples/photograph/{name}": f"examples/photograph/{name}"
        for name in (
            "item.jpg",
            "request.json",
            "response.json",
            "provenance.json",
            "README.md",
            "LICENSE.txt",
        )
    },
}
GENERATED = {"checksums.json", "publication.json"}
ARCHIVE_REVISION = "1b801c0a596d931a2fa2f587bf5e5eb70f096fea"
DROPPED_PREFIXES = ("evidence/", "runtime/")
LEGACY = re.compile(rb"veyra", re.IGNORECASE)
LEGACY_EXEMPT = {"manifest.json"}
WHEEL_MARKER = ("qev_runtime/frozen_identifiers.py", b'TRAINED_RESULT_MARKER = "VeyraResult:"')
COMPARISON_URL = "https://github.com/ken-jo/qev/tree/archive/pre-qev/reports/release-comparison"
PATCHED = {
    "release-comparison/README.md": (
        "- `input-protocol.json`: pinned data revisions, sample selection, shared input hash and\n"
        "  audit of the 15 released adaptation corpus snapshots.\n"
        "- `qev.json`, `laya-base.json`, `laya-specialist.json`: per-suite, type and workflow\n"
        "  results, truncation checks, serving precision and matched-hardware latency.\n",
        "- `qev.json`: per-suite, type and workflow results, truncation checks, serving\n"
        "  precision and matched-hardware latency.\n"
        "- `input-protocol.json` (pinned data revisions, sample selection, shared input hash and\n"
        "  corpus audit), `laya-base.json` and `laya-specialist.json` are in the\n"
        f"  [archived evidence]({COMPARISON_URL}).\n",
    )
}
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:\\\\|[A-Za-z]:\\|/home/|/mnt/|/Users/)")
LINK = re.compile(r"(\[[^\]]*\]\()([^)\s]+)(\))")
SOURCE_URL = "https://github.com/ken-jo/qev/blob/main/"


def package_links(text, name, source, root, packaged):
    """Point relative links whose target is not packaged at the source repository."""

    def replace(match):
        link = match.group(2)
        if "://" in link or link.startswith(("#", "mailto:")):
            return match.group(0)
        path, _, fragment = link.partition("#")
        if posixpath.normpath(posixpath.join(posixpath.dirname(name), path)) in packaged:
            return match.group(0)
        target = ((root / source).parent / path).resolve()
        if not target.exists() or not target.is_relative_to(root):
            raise ValueError(f"Broken link in {source}: {link}")
        url = SOURCE_URL + target.relative_to(root).as_posix()
        return match.group(1) + url + ("#" + fragment if fragment else "") + match.group(3)

    return LINK.sub(replace, text)


def dropped(name, path):
    if name in LEGACY_EXEMPT or name in CHECKPOINT_HASHES:
        return False
    return (
        name.startswith(DROPPED_PREFIXES)
        or LEGACY.search(name.encode())
        or LEGACY.search(path.read_bytes())
    )


def legacy_hits(folder):
    """Return packaged files that carry the legacy name outside the allowed exceptions."""
    hits = []
    for path in sorted(folder.rglob("*")):
        name = path.relative_to(folder).as_posix()
        if not path.is_file() or name in LEGACY_EXEMPT:
            continue
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    raw = archive.read(member)
                    if member == WHEEL_MARKER[0]:
                        raw = raw.replace(WHEEL_MARKER[1], b"", 1)
                    if LEGACY.search(member.encode()) or LEGACY.search(raw):
                        hits.append(name + "!" + member)
        elif LEGACY.search(name.encode()) or LEGACY.search(path.read_bytes()):
            hits.append(name)
    return hits


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--removed-list", type=Path, help="write the dropped paths here")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    if args.wheel.name != f"qev-{version}-py3-none-any.whl":
        raise ValueError("Wheel does not match the checkout version")
    if args.output.exists():
        raise FileExistsError(args.output)
    previous = json.loads((args.previous / "checksums.json").read_text("utf-8"))
    inherited = sorted(set(previous) - set(FROM_CHECKOUT) - GENERATED)
    for name in inherited:
        if digest(args.previous / name) != previous[name]:
            raise ValueError("Previous package file changed: " + name)
    removed = [n for n in inherited if dropped(n, args.previous / n)]
    carried = [n for n in inherited if n not in removed and n not in PATCHED]
    for name, expected in CHECKPOINT_HASHES.items():
        if previous.get(name) != expected:
            raise ValueError("Previous package does not hold the pinned checkpoint: " + name)
    with zipfile.ZipFile(args.wheel) as archive:
        names = archive.namelist()
        if any(n.split("/")[0].lower().startswith("veyra") for n in names):
            raise ValueError("Wheel still contains legacy runtime modules")
        for module in sorted((root / "src/qev_runtime").glob("*.py")):
            if archive.read("qev_runtime/" + module.name) != module.read_bytes():
                raise ValueError("Wheel runtime differs from the checkout")
    for name in carried:
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.previous / name, target)
    for name, (old, new) in PATCHED.items():
        text = (args.previous / name).read_text("utf-8")
        if text.count(old) != 1:
            raise ValueError("Unexpected previous content: " + name)
        (args.output / name).parent.mkdir(parents=True, exist_ok=True)
        (args.output / name).write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    packaged = (
        set(carried)
        | set(PATCHED)
        | set(FROM_CHECKOUT)
        | GENERATED
        | {"runtime/" + args.wheel.name}
    )
    for name, source in FROM_CHECKOUT.items():
        raw = (root / source).read_bytes()
        if source.endswith((".md", ".json", ".toml", ".lock")) and LOCAL_PATH.search(raw.decode()):
            raise ValueError("Local path found in " + source)
        if name.endswith(".md"):
            raw = package_links(raw.decode("utf-8"), name, source, root, packaged).encode("utf-8")
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    wheel_target = args.output / "runtime" / args.wheel.name
    if wheel_target.exists():
        raise FileExistsError("Previous package already holds this wheel version")
    wheel_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.wheel, wheel_target)
    hits = legacy_hits(args.output)
    if hits:
        raise ValueError("Legacy name left in package: " + ", ".join(hits))
    publication = {
        "project": "qev",
        "version": version,
        "model_version": "0.1.1",
        "runtime_wheel_file": "runtime/" + args.wheel.name,
        "runtime_package": "qev_runtime",
        "source_repository": "https://github.com/ken-jo/qev",
        "backbone": "Qwen/Qwen3.5-2B",
        "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "weights_sha256": digest(args.output / "head.safetensors"),
        "manifest_sha256": digest(args.output / "manifest.json"),
        "runtime_wheel_sha256": digest(wheel_target),
        "previous_package_checksums_sha256": digest(args.previous / "checksums.json"),
        "files_carried_byte_identical": len(carried),
        "previous_files_removed": len(removed),
        "removed_files_remain_at_revision": ARCHIVE_REVISION,
        "weights_and_calibration_unchanged": True,
        "separate_publication_verification": (
            "release/model-inference.json in the source repository"
        ),
    }
    (args.output / "publication.json").write_text(
        json.dumps(publication, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    checksums = {
        p.relative_to(args.output).as_posix(): digest(p)
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    }
    (args.output / "checksums.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({**publication, "files": len(checksums) + 1}))
    if args.removed_list:
        args.removed_list.write_text("\n".join(removed) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
