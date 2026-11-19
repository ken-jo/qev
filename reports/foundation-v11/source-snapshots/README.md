# Executed source snapshots

These text files preserve the exact Python bytes used to build the frozen
Foundation v11 corpus and feature cache. The working scripts were subsequently
formatted; `provenance.json` records both SHA-256 values and an equal Python AST.
They contain no dataset records, images, credentials or external source code.

The original `prepare_foundation_text.py` and `build_foundation_data.py` embed
their own source hashes in their outputs. Consequently a formatting-only change
changes those metadata hashes even though the generated examples are equivalent.
For exact historical reproduction, run these `.py.txt` files with Python, or copy
their bytes to a `.py` filename. Run from the repository root with Veyra installed.
The text selection tool used `pyarrow==25.0.1`; training did not require pyarrow.

The original feature specification also records source paths, so its protocol
hash can change if the script is executed under a different path. Compare the
dataset, weights, feature tensors and input settings as well as the protocol.
