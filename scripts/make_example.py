"""Create an original geometric image and a runnable single-question request."""

import json
from pathlib import Path

from veyra.benchmark import write_fixture

path = write_fixture(Path("data/example"))
request = json.loads(path.read_text(encoding="utf-8"))
request["questions"]["color"]["instructions"] = "What is the color of the leftmost object?"
path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
print(path.resolve())
