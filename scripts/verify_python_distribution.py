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
        if entrypoints["console_scripts"].get("qev-research") != "qev_runtime.research_cli:main":
            raise ValueError("Missing QEV research command")
        for package in ("qev", "qev_runtime", "qev_decision_index"):
            if f"{package}/py.typed" not in names:
                raise ValueError("Missing py.typed marker: " + package)
        modules = sorted((root / "src/qev_runtime").glob("*.py"))
        if len(modules) != 43:
            raise ValueError("Unexpected runtime module count")
        for module in modules:
            if archive.read("qev_runtime/" + module.name) != module.read_bytes():
                raise ValueError("Packaged runtime changed while packaging")
        public = sorted((root / "src/qev").glob("*.py"))
        if len(public) != 6:
            raise ValueError("Unexpected public module count")
        for module in public:
            if archive.read("qev/" + module.name) != module.read_bytes():
                raise ValueError("Public API changed while packaging")
        for asset in (root / "src/qev/playground").rglob("*"):
            if not asset.is_file() or "__pycache__" in asset.parts:
                continue
            name = "qev/playground/" + asset.relative_to(root / "src/qev/playground").as_posix()
            if name not in names or archive.read(name) != asset.read_bytes():
                raise ValueError("Packaged playground asset is missing or changed: " + name)
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
