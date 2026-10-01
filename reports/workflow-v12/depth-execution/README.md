# Full-depth execution comparison

This measurement used four declared original training inputs and **zero learning rates**.
It did not produce a trained checkpoint or use development, calibration or final predictions.

| Activation checkpointing | 265-token text | 693-token text | 383-token text | 375-token photo | Largest allocated memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 0.252 s | Allocator limit | 0.308 s | 0.342 s | 6,787 MiB on completed cases |
| Earlier 12 layers | 0.345 s | 0.509 s | 0.355 s | 0.420 s | 6,604 MiB |
| All 24 layers | 0.420 s | 0.605 s | 0.457 s | 0.525 s | 4,627 MiB |

Times are medians of three synchronized training steps after one warmup on RTX 4060 Ti 8GB.
The steps include processing, forward/backward, clipping and zero-rate AdamW updates. Parameter
hashing and the saved gradient copy are outside the timed repetitions. These are **training
times**, not the HTTP inference latency required for release. The allocator was limited to 85%
of device memory. The uncheckpointed 693-token case exceeded that limit; other system/device
conditions and unmeasured input shapes may differ.

Every completed comparison had exactly equal logits and saved gradients. On the 693-token
case this only compares earlier-12 against all-24, because the uncheckpointed run did not finish.
All parameter values and original inputs stayed unchanged. AdamW state was actually allocated.
Gradient tensors stay local; the public JSON contains aggregate results and their hashes.

The prospectively declared selection rule chose earlier-12, the fastest method that completed
all four cases. The [next learning study](../../../configs/workflow-depth-study-v12.json) uses
this method through 693 tokens and full checkpointing for longer inputs. The longer-input
fallback is a memory precaution, not a measured whole-corpus memory guarantee.

## Reproduction

- Declaration: `configs/workflow-depth-execution-v12.json`
- Command: `.venv/Scripts/python.exe -u scripts/profile_workflow_depth.py`
- Original local output: `runs/workflow-v12-depth-execution`
- Public protocol/results are exact copies; `manifest.json` binds their bytes.
- Outputs are immutable and the command refuses to overwrite them.

No accuracy improvement, seed robustness or release eligibility is established by this probe.
