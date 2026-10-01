# Final QEV / LAYA comparison evidence

The comparison was run on 2026-10-01 with frozen checkpoints on an RTX 4060 Ti 8 GB.
All three models saw 2,000 official typed decisions and the same first 400 test examples
from each of AG News and DAIR Emotion. No training or calibration was performed.

- [Narrative and complete comparison tables](https://github.com/ken-jo/qev/blob/main/docs/LAYA_COMPARISON.md).
- `input-protocol.json`: pinned data revisions, sample selection, shared input hash and
  audit of the 15 released adaptation corpus snapshots.
- `qev.json`, `laya-base.json`, `laya-specialist.json`: per-suite, type and workflow
  results, truncation checks, serving precision and matched-hardware latency.
- `comparison.json`: prediction hashes, identity/target equality checks and paired
  group-bootstrap accuracy intervals.
- `accuracy.png` / `accuracy.svg`: figures generated from those aggregate reports.

Reproduction scripts are `scripts/prepare_release_comparison.py`,
`scripts/evaluate_release_comparison.py`, `scripts/summarize_release_comparison.py`
and `scripts/plot_release_comparison.py`. The preparation script needs PyArrow; plotting
needs Matplotlib. Evaluation uses the pinned QEV runtime and `laya==0.3.20`.
Use each script's `--help` for paths. `evaluate_release_comparison.py` requires Python
isolated mode (`python -I`) and runs one model per process. Prepare the LAYA checkpoints
with the recorded revisions and the reference inventory in `reports/foundation-v11`.
Original benchmark texts are fetched from upstream and not redistributed here.

The request hash is `585850d575d9dbdf943683936dda7e07b9e5525b60875906926e9915c557d1d6`.
The benchmark wheel was frozen before the model-card rewrite; its 38 inference modules
and learned weights match the final release. Later packaging verification checks that
identity. Changes to wheel metadata do not imply another accuracy experiment.

AG News and Emotion are task-held-out relative to the audited QEV adaptation sources.
Unknown backbone pretraining overlap is not ruled out. Typed-decisions is an already
inspected, adapted regression benchmark. These scopes must remain separate in claims.
