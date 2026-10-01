# Model size and tensor precision

QEV is an independent decision model adaptation using the pinned Qwen3.5-2B backbone.
The QEV name does not change the architecture or upstream weight provenance.

## Parameter counts

| Component | Parameters |
| --- | ---: |
| Vision backbone | 331,416,576 |
| Language backbone | 1,881,825,088 |
| Backbone used by QEV | 2,213,241,664 |
| Stored language LoRA | 7,815,168 |
| Option readout | 32,768 |
| Condition readout | 6,144 |
| Condition binding head | 138,304 |
| Stored adaptation/readout total | 7,992,384 |
| Model with unmerged adapters | 2,221,234,048 |
| Model after LoRA merge | 2,213,418,880 |

Counts come from the pinned `Qwen3_5Model` parameter structure and safetensors headers,
without allocating a second backbone on the GPU. LoRA weights merge into existing
backbone matrices, so they are not additional inference parameters after merging.

## Dtypes and files

| Item | Precision / format | Size |
| --- | --- | ---: |
| QEV head checkpoint | safetensors; 307 tensors, all FP32 | 32,009,800 bytes (32.01 MB) |
| Upstream checkpoint file | safetensors; mostly BF16, 2,592 FP32 elements | 4,548,221,488 bytes (4.55 GB) |
| GPU inference backbone | BF16 | Hardware memory use also includes activations and runtime buffers |
| GPU decision readouts / binding | FP32 | Probabilities are computed through the existing calibrated runtime |
| CPU inference | FP32 | Higher weight memory use than the BF16 GPU path |

The upstream file contains 2,274,069,824 tensor elements, including 60,828,160 MTP
elements that the QEV encoder does not use. This explains the difference between the
download's tensor count and the 2.213B backbone used for inference. QEV does not ship
an INT8 or INT4 quantized checkpoint.

The Python wheel is code, not the complete neural network. `qev download` retrieves
the 32 MB decision checkpoint and the approximately 4.55 GB upstream weights separately.
Historical evidence in the complete release ZIP makes that ZIP larger than the primary
head checkpoint; these are different download sizes.

Frozen base revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`.
Frozen head SHA-256: `84b57aeeb987f73416ac0f796150957d1c5beb1ed373733053e46438f77a8fee`.
