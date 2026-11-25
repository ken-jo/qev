# Bundled option checkpoints

Format 3 packages an inference-only, dense Qwen3.5 backbone inside an option
checkpoint. Format 2 serialization and loading are unchanged.

The package root contains `manifest.json` and `head.safetensors`. The manifest's
optional `backbone.layout` selects where config, safetensors weights, tokenizer
and processor files live:

- `"root"`: backbone files live beside the manifest, including `config.json`,
  `model.safetensors`, tokenizer/processor files, `chat_template.jinja` and
  `vocab-map.json` when present. Recommended for new releases: this follows the
  Hub convention for single-backbone models and places `config.json` at the root,
  where the Hub counts downloads for libraries without a registered counter.
- `"subdir"`: backbone files live in `backbone/`. Omitting `layout` means
  `"subdir"`, so existing manifests and pinned releases keep loading unchanged.

Unknown layout values and packages mixing both layouts are rejected.
Root-layout packages need `qev` at or after
`9f3379cbcca6a63d43bd9cb9092f93d9e8e6371c`; earlier versions load only `"subdir"`.
The manifest retains the usual encoder, readout, calibration and completed-training
fields and declares:

```json
{
  "format_version": 3,
  "architecture": "option_readout",
  "backbone": {
    "bundled": true,
    "layout": "root",
    "model_id": "organization/source-model",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "depth": 16,
    "vocabulary": 32768,
    "files": {
      "config.json": "<64 lowercase hex SHA-256 characters>",
      "model.safetensors": "<64 lowercase hex SHA-256 characters>",
      "tokenizer.json": "<64 lowercase hex SHA-256 characters>",
      "tokenizer_config.json": "<64 lowercase hex SHA-256 characters>",
      "processor_config.json": "<64 lowercase hex SHA-256 characters>"
    }
  },
  "adaptation": {},
  "head_key_prefix": "",
  "weights_sha256": "<SHA-256 of head.safetensors>",
  "modes": {"accurate": {"image_max_pixels": 1638400}},
  "default_mode": "accurate"
}
```

This example omits the usual readout, calibration and training fields. Every
backbone file, including chat templates and vocabulary maps when present,
must be listed relative to its layout's backbone directory. For root layout,
the inventory is exactly every non-directory path under the package root minus
the fixed package-file allowlist: `manifest.json`, `head.safetensors`, `README.md`,
`LICENSE`, `.gitattributes`. These names cannot appear in a root backbone inventory.
Any other unlisted file, including files in subdirectories, is rejected. A root
package cannot contain a `backbone/` directory; a subdir package cannot contain
stray root files outside that same package-file allowlist.

Both layouts ignore exactly the real top-level `.git/` directory and the real
`.cache/huggingface/` directory, including their contents. They are never hashed
or loaded. Symlinks to directories are rejected, including at these locations;
`.cache/other`, a file named `.cache`, or a file named `.git` receives no exclusion.
Directory symlinks elsewhere in the inventory are also rejected.

Sharded safetensors use a hashed `model.safetensors.index.json`
whose shard set matches the inventory. Missing files, unlisted files, changed
hashes, incompatible depth/vocabulary and incomplete weights are errors.
Tokenizer/processor configuration file references, including versioned
`fast_tokenizer_files`, must be normalized relative names in that verified
inventory. Absolute paths, traversal and unlisted targets are rejected before
the processor loads; file symlinks continue to be checked by their target bytes.
All raw safetensors headers must contain exactly the dense model's tensor names
and shapes, including across shards. Duplicate keys, incorrect index mappings
and extra tensors that Transformers would normally ignore are errors.
The model's multimodal token IDs must match the loaded tokenizer and processor
and fit within its embedding vocabulary.
The manifest is the trust root; hashes detect corruption relative to it.

`model_id` and `revision` identify the source, not a download target. Loading uses
only verified local bytes, even if `local_files_only=False` was supplied. Remote
code, pickle weights, live adapters and early-exit presets are unsupported.
Hub snapshot file symlinks are supported and their target bytes are verified.

`head_key_prefix` may be empty (the default) or `head.`. The latter strips the
prefix from all tensor keys in memory after checking the original file hash,
allowing a frozen training export to ship without rewriting its bytes.

`OptionModel.load(directory, mode="accurate")` and the existing
`qev_decision_index:QEVEngine` work with this format. The engine accepts either
a local package directory or a Hub repository id and revision. `fast` mode is
rejected with the available mode listed. Loaded bundles cannot be saved through
`OptionModel.save`; package dense files and their inventory explicitly.

To copy an existing subdir package into a new root package without changing the
source, head bytes or backbone hashes, run from the source checkout:

```sh
PYTHONPATH=src python scripts/flatten_bundled_checkpoint.py SRC DST
```

The helper refuses to overwrite any destination. It verifies both inventories,
compares the manifest (only `backbone.layout` changes), and checks the head and
package-file hashes. Hub file symlinks are copied as their verified target bytes;
local Git and download metadata are not copied. To also load both packages on
CPU and require bit-identical answers and probabilities for N deterministic text
requests, add `--compare N`, for example:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/flatten_bundled_checkpoint.py SRC DST --compare 3
```

The JSON result reports the request count and maximum absolute probability
difference, which must be `0.0`.
