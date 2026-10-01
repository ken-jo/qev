"""Download the versioned GitHub checkpoint asset and verify its SHA-256."""

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    url = "https://github.com/ken-jo/qev/releases/download/v0.1.1/qev-0.1.1.zip"
    target = args.output.resolve()
    checkpoint = target / "qev"
    if checkpoint.exists():
        raise FileExistsError("Checkpoint destination already exists")
    target.mkdir(parents=True, exist_ok=True)
    archive_path = target / "qev-0.1.1.zip"
    urllib.request.urlretrieve(url, archive_path)
    with urllib.request.urlopen(url + ".sha256", timeout=60) as response:
        expected = response.read().decode().split()[0]
    with archive_path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Downloaded archive checksum mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            path = (target / name).resolve()
            if not path.is_relative_to(checkpoint):
                raise ValueError("Unexpected archive member")
        archive.extractall(target)
    print(checkpoint)


if __name__ == "__main__":
    main()
