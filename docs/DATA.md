# Public dataset

`qev-data` packages 15 historical corpus configurations, not 15 disjoint datasets.
Each archive contains original eligible JSONL records, referenced image bytes and a
publication manifest. Viewer JSONL exposes the same requests and targets as JSON strings
to avoid incompatible nested candidate schemas. Original train/dev/calibration/test values
are preserved. The release manifest records each original and exported file checksum.

Public corpus files use `qev-<stage>.zip`. Their extracted stage
folders and original record/source IDs preserve the recorded training provenance.
For example, `qev-starter-v2.zip` extracts to `starter-v2/`.

## Attribution and licenses

| Source | License | Conversion |
| --- | --- | --- |
| Authored diagrams, policy rules, conditional probabilities and workflows | Apache-2.0 | Generated evidence, questions and probability targets |
| [Beans, AIR Lab at Makerere](https://github.com/AI-Lab-Makerere/ibean) | MIT | Typed questions over leaf photos |
| [TrashNet, Gary Thung and Mindy Yang](https://github.com/garythung/trashnet) | MIT | Recognition and request-defined policy views; Kaggle version 1 source matched by hashes |
| [SNLI, Stanford NLP / Bowman et al.](https://nlp.stanford.edu/projects/snli/) | CC-BY-SA-4.0 | Premise/hypothesis decisions and policy views; derived material remains share-alike |
| [BANKING77, PolyAI / Casanueva et al.](https://github.com/PolyAI-LDN/task-specific-datasets) | CC-BY-4.0 | Sampled candidate intent and policy views |
| [LocalLLaMA typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 | Recorded training partition and teacher-agreement distributions |

The dataset is a collection under multiple licenses. The repository's Apache license does
not relicense CC-BY or CC-BY-SA records. Each record carries its source/license/revision;
original MIT notices and CC legal texts are included. Original source observations were
selected and converted into the recorded typed request format; these are modified datasets.

**CIFAR-10:** the [upstream card](https://huggingface.co/datasets/uoft-cs/cifar10) marks the
license unknown. Its evaluation images and requests are excluded. Exclusion manifests retain
IDs, hashes, split and source references for direct upstream reconstruction. No selected
training rows are removed by this license exclusion. The model repository contains no raw
training corpus or third-party evaluation photos.

## Integrity and use

`scripts/package_public_data.py` fails on duplicate IDs, group leakage, byte-identical image
leakage between splits, incorrect image hashes, paths outside a corpus or unexpected licenses.
These are checks within each corpus. They do not establish cross-stage disjointness, distinct
real objects, or absence from Qwen pretraining.

```sh
python scripts/package_public_data.py --source /path/to/original-workspace --output dist/qev-data
```

This maintainer command requires the recorded source snapshots. Users can download the
published archives directly; they do not need to recreate the original private workspace.
The repository includes the public dataset manifest under `release/`.
