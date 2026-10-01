"""Create versioned GitHub archives using only audited package allowlists."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for kind, folder, name in (
        ("model", args.model, "qev"),
        ("dataset", args.data, "qev-data"),
    ):
        checks = json.loads((folder / "checksums.json").read_text("utf-8"))
        destination = args.output / f"{name}-0.1.1.zip"
        with zipfile.ZipFile(destination, "x") as archive:
            for relative in sorted([*checks, "checksums.json"]):
                path = (folder / relative).resolve()
                if not path.is_relative_to(folder.resolve()):
                    raise ValueError("Unexpected archive input")
                raw = path.read_bytes()
                if relative in checks and hashlib.sha256(raw).hexdigest() != checks[relative]:
                    raise ValueError("Package changed since audit")
                compression = zipfile.ZIP_STORED if path.suffix == ".zip" else zipfile.ZIP_DEFLATED
                archive.writestr(name + "/" + relative, raw, compress_type=compression)
        with destination.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        destination.with_suffix(".zip.sha256").write_text(
            f"{sha}  {destination.name}\n", encoding="utf-8"
        )
        outputs[kind] = {
            "file": destination.name,
            "sha256": sha,
            "bytes": destination.stat().st_size,
        }
        print(json.dumps({kind: outputs[kind]}), flush=True)
    (args.output / "assets.json").write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
