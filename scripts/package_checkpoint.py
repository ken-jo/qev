"""Package only a verified trained head, its manifest, card, and license notices."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from veyra.checkpoint import load_head
from veyra.release_gate import evidence_matches

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--name", default="veyra-starter-v2")
parser.add_argument("--acceptance-audit", type=Path)
args = parser.parse_args()
if not args.name or any(char in args.name for char in ("/", "\\", ":")) or args.name in {".", ".."}:
    raise ValueError("archive name must be one safe directory component")
manifest = json.loads((args.checkpoint / "manifest.json").read_text(encoding="utf-8"))
evidence_assets = {}
if manifest.get("architecture") == "option_readout":
    if not args.acceptance_audit:
        raise ValueError("option checkpoint publication requires a final acceptance audit")
    audit = json.loads(args.acceptance_audit.read_text())
    if (
        audit.get("passed") is not True
        or audit.get("checkpoint_weights_sha256") != manifest["weights_sha256"]
        or audit.get("checkpoint_manifest_sha256")
        != hashlib.sha256((args.checkpoint / "manifest.json").read_bytes()).hexdigest()
    ):
        raise ValueError("acceptance audit does not verify this checkpoint")
    evidence_assets["reports/acceptance.json"] = args.acceptance_audit
    for name in ("quality", "calibration", "integration", "regression", "protocol"):
        evidence = audit["evidence"][name]
        path = Path(evidence["path"])
        if not evidence_matches(path.read_bytes(), evidence):
            raise ValueError("acceptance evidence changed after its audit")
        evidence_assets[f"reports/{name}.json"] = path
    from veyra.option_model import OptionModel

    model = OptionModel.load(args.checkpoint, local_files_only=True)
    del model
else:
    load_head(args.checkpoint)
if args.output.exists():
    raise FileExistsError("use a new archive path")
assets = {
    **evidence_assets,
    "head.safetensors": args.checkpoint / "head.safetensors",
    "manifest.json": args.checkpoint / "manifest.json",
    "MODEL_CARD.md": Path("MODEL_CARD.md"),
    "LICENSE": Path("LICENSE"),
    "NOTICE": Path("NOTICE"),
    "beans-MIT.txt": Path("docs/data-licenses/beans-MIT.txt"),
}
if not all(path.is_file() for path in assets.values()):
    raise FileNotFoundError("release assets are incomplete")
args.output.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(args.output, "x", ZIP_DEFLATED) as archive:
    for name, path in assets.items():
        archive.write(path, f"{args.name}/{name}")
digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
args.output.with_suffix(".zip.sha256").write_text(
    f"{digest}  {args.output.name}\n", encoding="utf-8", newline="\n"
)
print(f"{digest}  {args.output.name} ({args.output.stat().st_size} bytes)")
