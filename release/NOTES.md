# Qwen3.5 Classification 0.1.0

Final research release of the Qwen-based Veyra line, renamed **Qwen3.5 Classification**.
The base is **Qwen/Qwen3.5-2B** at a pinned revision, with learned language LoRA and
typed decision readouts. The vision encoder remains frozen. This is not a new base
network or a JEV implementation.

## Downloads

- [Hugging Face model](https://huggingface.co/ken-jo/qwen3.5-classification)
- [Hugging Face dataset](https://huggingface.co/datasets/ken-jo/qwen3.5-classification-data)
- `qwen3.5-classification-0.1.0.zip`: trained checkpoint, runtime wheel, English model card and acceptance evidence.
- `qwen3.5-classification-data-0.1.0.zip`: 15 licensed historical corpus configurations, images, source attribution,
  split/hash manifests and viewer records. Configurations overlap; do not concatenate them.
- Each asset has a SHA-256 sidecar. Qwen base weights are downloaded separately.

## Results and limits

Fresh authored workflow accuracy: 69.38%. Fresh CIFAR-10 guard: 95.83%; SNLI: 87.78%;
sampled eight-candidate BANKING77: 85.50%. Resident serial local-photo HTTP p95: 114.94 ms
on RTX 4060 Ti 8 GB, excluding model loading, WAN and concurrency.

Overall final ECE: 12.58%; accepted uncertain-request expected error: 45.57%. Exploratory
2048 study: zero wins in 72 games. Image board-cell diagnosis: 5/28. These limitations
are part of the release; no general business-reliability or JEV/LAYA parity claim is made.

## Publication verification

163 tests passed. The 38 evaluated runtime modules, model weights and calibration were
preserved byte-for-byte. Actual packaged GPU inference passed for text and one photograph
covering choice, score and noul; answer probabilities exactly matched the earlier package
for those fixtures. All dataset archive/image hashes and source-license filters passed.
This reused the measured Python environment and cached Qwen base, not a clean-machine install.

Personal Korean notes and the old development history are archived privately. Public docs
are English; generated Korean training examples and the inherited localized playground remain.
CIFAR-10 raw evaluation images are excluded because redistribution rights are unresolved.

Maintainer: [GitHub](https://github.com/ken-jo) ·
[LinkedIn](https://www.linkedin.com/in/ik-chan-jo).
