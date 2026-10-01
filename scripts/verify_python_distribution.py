"""Inspect the wheel and sdist without importing the model or installing dependencies."""

import argparse
import configparser
import hashlib
import json
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]
    version = project["version"]
    wheel = args.directory / f"qev-{version}-py3-none-any.whl"
    source = args.directory / f"qev-{version}.tar.gz"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if any(
            PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or name.endswith((".safetensors", ".pt", ".pth", ".pyc"))
            for name in names
        ):
            raise ValueError("Unexpected artifact inside the Python wheel")
        metadata = BytesParser().parsebytes(archive.read(f"qev-{version}.dist-info/METADATA"))
        if metadata["Name"] != "qev" or metadata["Version"] != version:
            raise ValueError("Distribution identity differs from pyproject.toml")
        if metadata["License-Expression"] != "Apache-2.0":
            raise ValueError("Missing license metadata")
        entrypoints = configparser.ConfigParser()
        entrypoints.read_string(archive.read(f"qev-{version}.dist-info/entry_points.txt").decode())
        if entrypoints["console_scripts"]["qev"] != "qev.cli:main":
            raise ValueError("Missing QEV command")
        modules = sorted((root / "src/veyra").glob("*.py"))
        if len(modules) != 38:
            raise ValueError("Unexpected frozen runtime module count")
        for module in modules:
            if archive.read("veyra/" + module.name) != module.read_bytes():
                raise ValueError("Frozen runtime changed while packaging")
        for module in (root / "src/qev").glob("*.py"):
            if archive.read("qev/" + module.name) != module.read_bytes():
                raise ValueError("Public API changed while packaging")
    with tarfile.open(source, "r:gz") as archive:
        names = archive.getnames()
        prefix = f"qev-{version}/"
        if any(not name.startswith(prefix) or ".." in PurePosixPath(name).parts for name in names):
            raise ValueError("Unexpected source archive path")
        for required in ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "src/qev/cli.py"):
            if prefix + required not in names:
                raise ValueError("Source distribution is missing " + required)
    print(
        json.dumps(
            {
                "passed": True,
                "project": "qev",
                "version": version,
                "frozen_modules": len(modules),
                "distributions": [
                    {
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in (wheel, source)
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
