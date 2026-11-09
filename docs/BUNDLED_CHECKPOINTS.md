# Bundled option checkpoints

Format 3 packages an inference-only, dense Qwen3.5 backbone inside an option
checkpoint. Format 2 serialization and loading are unchanged.

The package root contains `manifest.json`, `head.safetensors`, and a `backbone/`
directory holding config, safetensors weights, tokenizer and processor files.
The manifest retains the usual encoder, readout, calibration and completed-training
fields and declares:

```json
{
  "format_version": 3,
  "architecture": "option_readout",
  "backbone": {
    "bundled": true,
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
file in `backbone/`, including chat templates and vocabulary maps when present,
must be listed. Sharded safetensors use a hashed `model.safetensors.index.json`
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
