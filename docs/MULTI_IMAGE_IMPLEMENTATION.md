# QEV multi-image implementation status — 2026-10-09

## Location and state

- Checkout: `C:/tmp/qev-multi-image-20261009`.
- Branch: `feat/multi-image-input`.
- Base: `bdd57b6346581be3d8a00089fc34eb10ce0931ff`, including merged QEV PRs 1 and 2.
- Implementation is local and uncommitted. No package installation, commit, push,
  publication, experiment registration, training or inference was performed.
- `C:/tmp/vision-qev` has no source/package changes relative to its existing HEAD;
  its previously present roadmap/research changes were retained. The active study's
  source, environment, processes, data and checkpoints were not modified.

## Implemented

1. Strict SDK/HTTP schema accepts zero to four images with the existing image-path format.
2. Both Qwen encoder paths process every image separately in list order. Multi-image
   prompts label visual placeholders `Image 1:`, `Image 2:`, etc. Every option-model
   question/view receives the full ordered group. No collage or image selection is used.
3. Zero/one-image prompt formatting and encoder configuration remain structurally the
   same. Learned weights, backbone revision, calibration coefficients and training
   recipes were not edited. Numerical parity has not been measured.
4. Shared image handling checks every resolved path, source size and decoded geometry.
   All opened images are closed on normal completion and on a later-image/processor error.
5. The existing complete-request processed-token cap is enforced before backbone
   execution. Multi-image input does not bypass the cap or silently discard evidence.
6. Playground adds Image 2-4 slots, rejects gaps and clears extra images on preset changes.
   The prior positional prediction arguments and single-image filename are retained.
7. Replay and refresh overlap checks audit all images. Ordered multi-image fingerprints
   include every image; the historical zero/one-image task fingerprint format is retained.
8. Responses expose `usage.input_images`. Source guide, agent contract and a two-image
   reference/current request example document the extension and its qualification limits.

## Static checks completed

- Python AST parsing: 10 changed/new Python files, no syntax errors.
- Ruff E/F/I checks on those 10 files: passed, with caching disabled.
- `git diff --check`: passed.
- Example JSON and project TOML: parse successfully.
- Source scan: no remaining `state.images[0]` assumption in the runtime/playground.
- Read-only inspection of installed Transformers 5.17 maps `qwen3_5` to
  `Qwen3VLProcessor`; visual placeholders are resolved with ordered image-grid indices.
- No tests were added or executed. These static checks do not establish native runtime
  correctness, browser interaction, probability accuracy or speed.

## Remaining verification and activation

Keep the existing training execution on its admitted source/environment. Native model
verification and installation remain pending; the published SDK 0.2.1 still accepts
one image. Multi-image accuracy/calibration and QEV's **100 ms or less** target are
unmeasured. Vega's separate 150 ms requirement does not apply to QEV.

The next verification should establish:

- Zero/one-image numerical parity with the existing runtime on an identical workload.
- Native two/four-image processing, including multiple questions/views, without missing
  or reordering image evidence; confirm actual processor image counts and token accounting.
- Rejection of a fifth image, root escapes, missing/oversized later images and token
  overflow before model execution; successful cleanup after each failure.
- Playground input order, preset reset and request/response behavior.
- Every-image exclusion in replay/refresh audits, including overlap only in a later image.
- Measured reference/current and multi-view decisions, probability quality, errors and
  coverage, plus latency of **100 ms or less** on the intended device/workload.
  Extra evidence does not guarantee
  better decisions, and historical single-image results do not qualify multi-image use.

Only after verification should the source extension be installed into a separate runtime
or considered for a versioned release. No scheduled monitor was changed to do that work.
