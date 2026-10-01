# Visual 2048: first exploratory runs

The playground now renders a 2048 board, sends that exact PNG to the resident Veyra v13
checkpoint, executes its chosen direction, and records the actual resulting game state.
No model training or checkpoint changes were made. This is an executable image/action demo;
the current model did **not** demonstrate competent 2048 play.

## Protocol

- Real local HTTP calls on an RTX 4060 Ti; one image, one `choice` question, all four directions.
- Fixed `visual-2048-v1` prompt with rules and the last six executed direction names. No board
  array, score, legal-direction filtering, planner, or heuristic action enters the model input.
- The game engine implements merging, tile spawning and scoring. It supplies pixels to the
  model and independently determines whether the selected move changed the board.
- Before every decision the runner downloads the visible frame. Its SHA-256 must match the
  PNG actually passed to inference. All **44 initial decisions** matched (43 exploration
  decisions plus one policy-respecting decision).
- Each exploration requested up to 60 decisions. Stop at 2048, game over, or four consecutive
  unchanged actions. None reached 2048 or the 60-decision budget; all stalled.
- The default checkbox respects the unchanged model's abstention. The first seed-11 run
  abstained immediately: zero executed moves. That first inference took 500.70 ms.
- The four subsequent runs explicitly used `--execute-abstained` to observe recommendations
  in this isolated game. **31 of 43** directions were marked abstained. These are not accepted
  policy decisions, and their exploration does not demonstrate improved calibration.
- No warmup was discarded. Per-run medians below cover all decisions in that run. Model
  timing includes model input preparation/forward; it excludes board rendering, HTTP, UI,
  model loading and evidence-file writes. Browser interaction automation was unavailable;
  these loops were executed through the same live endpoints used by the page.

## Observed results

| Seed | Source PNG | Theme | Game score | Largest tile | Valid / executed moves | Abstained / decisions | Model median |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 512×512 | Classic | 4 | 4 | 3 / 7 | 7 / 7 | 127.34 ms |
| 29 | 512×512 | Classic | 48 | 8 | 16 / 20 | 15 / 20 | 137.98 ms |
| 47 | 256×256 | Classic | 20 | 8 | 6 / 10 | 3 / 10 | 96.30 ms |
| 83 | 512×512 | Monochrome contrast | 4 | 4 | 2 / 6 | 6 / 6 | 132.20 ms |

All four runs selected the same unproductive direction four times before stopping. Across
them, 27 of 43 executed actions changed the board; **this is a valid-move rate, not strategic
accuracy**. Early boards allow many easy valid moves. There is no established gameplay or
computer-use success rate here, and a 60-step ceiling is not a fair test of reaching 2048.

Seeds differ across size/theme settings, so these rows cannot establish a resolution or theme
effect. They are tiny exploratory samples, not independent evidence of generalization. There
is no matched JEV/Astra arm, equal task budget or comparable cost accounting.

After the final server restart, a seed-29 rerun reproduced the same 20 directions, score 48,
largest tile 8, 16 valid moves and four-unchanged-action stop. All 20 input hashes matched.
That separate integration check had a 140.47 ms model median, 150.67 ms local HTTP median,
and a 507.20 ms first inference after loading; it is not included in the initial table or
44-decision aggregate. Its local evidence is `runs/visual2048/seed29-final-demo/`.
The page and both new assets returned HTTP 200 on localhost and the host's Tailscale IP.
Python compilation, focused Ruff checks and JavaScript syntax checks passed; interactive
browser behavior remains unverified because the browser automation runtime failed to start.

## Reproduce and inspect

Start the playground, then use a fresh output directory:

```powershell
.venv/Scripts/python.exe scripts/run_visual_2048.py --seed 11 --steps 30 --output runs/visual2048/policy-new
.venv/Scripts/python.exe scripts/run_visual_2048.py --seed 11 --steps 60 --execute-abstained --output runs/visual2048/explore-new
```

Repeat with seeds 29, 47 (`--size 256`), and 83 (`--theme contrast`). The runner preserves every
pre-action PNG, the final PNG, the exact model request/response and both board states. Records
are in `runs/visual2048/seed*-*/`; public aggregates and per-step decisions are in
[initial-runs.json](initial-runs.json). Local raw artifacts are intentionally outside Git.

## Next model work

The confirmed failure is repeated non-progress despite fresh board images. These runs alone
do not identify whether tile recognition, spatial reasoning, long-term action value, recent
action imitation, or a combination is responsible.

1. Diagnose tile/position reading and whether a move changes a board with separately scored
   visual questions; compare against an image-blind control on the same states.
2. Build supervised screenshot/action trajectories using an independent game solver or human
   demonstrations. Split by complete trajectories/board families; reserve unseen themes,
   fonts and resolutions. Keep internal board state out of inference requests.
3. Evaluate held-out episodes with equal move budgets against simple fixed/random policies
   and a solver. Measure max tile, score, stalls, illegal/no-op actions and real episode time.
4. Recalibrate abstention on separate gameplay data; require improvement in task success and
   false-confident no-op rate, while retaining the existing text/photo regression gates.

No gameplay fine-tuning or performance claim is hidden in the demo implementation.

## Text-only follow-up

[LAYA and Veyra text-board runs](../text2048/README.md) use the exact numeric board as text.
They isolate a text-input application from this image-input experiment. That first text
protocol is our own harness, not a reproduction of the original `jev-use` demonstration.
