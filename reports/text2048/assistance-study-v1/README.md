# Detailed instructions, assistance and image diagnosis

The exact v13 checkpoint was evaluated in 72 exploratory games: 24 development games
and 48 games on eight new seeds. There were 8,423 actions, 7,002 game model calls and five
single-legal-direction engine steps. **Zero games reached 2048.**

| New-seed profile | Mean game score | Highest tile |
| --- | ---: | ---: |
| Detailed text, no engine assistance | 2.5 | 4 |
| Detailed raw image, no engine assistance | 2.5 | 4 |
| Legal directions supplied | 1,526 | 256 |
| Next board and empty-cell counts supplied | 1,983 | 512 |
| Next board, empty counts and immediate score supplied | 3,027 | 512 |
| Random legal direction control | 1,037 | 128 |

The detailed instructions are in [assisted-instructions.txt](assisted-instructions.txt).
Raw images contain only the rendered board; the engine-assistance variants provide more
input information and are not pure image reasoning. No search-rank answer was supplied.
Abstained suggestions were executed in the exploratory runs and remain marked as abstained.

The independent board-cell readout diagnostic used 28 balanced seven-choice cases: image
5/28, text 21/28. Image requests abstained 26/28 times; text abstained 13/28. This narrow
position/digit binding test does not measure general photo recognition.

[summary.json](summary.json) contains the original protocol and aggregates;
[audit.json](audit.json) records deterministic replay and input-boundary checks. Their
historical pre-publication status strings and hashes are preserved unchanged. The audit
replayed all 8,423 actions and verified rendered input hashes for 86 game frames plus 28
diagnostic images. No untouched-Qwen ablation was performed.

The failed result motivates separate future architecture work. It is not used to tune
this frozen release or revise its earlier acceptance thresholds.
