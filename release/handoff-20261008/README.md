# QEV handoff verification, 2026-10-08

This follow-up verifies the source changes in GitHub PR #1. It supplements the
historical SDK release evidence; it does not rewrite that evidence.

## Accepted changes

- Jev/SystemOne-style text and business-JSON requests retain the supplied candidates.
- Native `text`/`images` objects retain strict validation. Mixed image objects with
  unknown fields fail with HTTP 422 instead of silently dropping the photograph.
- Documentation identifies compatibility as unreleased and source-only. Published
  SDK 0.2.1 and its attached wheel still use the original native request format.
- Playground controls use English for every Gradio-supported browser locale, while
  the evidence remains multilingual. English strings are attributed in `NOTICE`.
- GitHub CI runs the complete CPU suite and lint checks for the changed source files.

## Actual verification

- Five unsafe mixed-state cases reproduced the old HTTP 200 behavior before the fix.
- The bounded schema/server regression suite passed 41 tests after the fix.
- The complete CPU suite passed 223 tests, with one existing Starlette/httpx warning.
  The final native stdout is in `cpu-tests.txt`; commands and hashes are in
  `verification.json`.
- `verify_ui.cjs` rendered the real Gradio component tree with `UIOnlyEngine`, which
  raises if inference is invoked. It checked all seven presets, all six licensed
  sample photographs, three decision types, Korean/Japanese/German browser locales,
  and a 390-pixel mobile viewport. No horizontal overflow or browser errors occurred.
- The desktop, selected-photo and mobile screenshots were visually inspected.
  The photographs retain the TrashNet MIT attribution shown in the interface and
  `src/qev/playground/samples/LICENSE.txt`.

The browser harness initially needed corrections for an already-selected radio
and for SVG icons being mistaken for the selected photograph. The final harness
waits for the exact selected photograph filename and all assertions passed.

## Scope

No model weights were loaded, no model inference was run, and no new accuracy or
latency was measured. The active distillation job, protected original QEV checkout,
Vega source, ledger and artifacts were left unchanged. Vega issue tasks T5/T6/T7
remain on hold. These source changes have not been released to PyPI or Hugging Face
as a new SDK. Hugging Face documentation publication has separate provenance.

`ui_preview.py` is a model-free preview helper. `extract_english_ui.cjs` records the
one-time extraction from the pinned Gradio frontend; the application uses the owned
JSON resource and does not import or parse private frontend assets at runtime.

## HF documentation publication

After PR #1 merged as `5e5cf9b310ead30d3fe6b72d4d59508e5c2545b3`, the model card
and its README checksum were published in one parent-guarded HF commit,
`c4503db4d09be56c6a7bf73b7fb7a356d9850deb`. Anonymous downloads match the merged
source. The complete 210-file tree retained the same paths; only `README.md` and
`checksums.json` changed. All other 208 file identities and all other 207 checksum
entries were preserved. See `hf-docs-publication.json` for exact hashes and inherited
historical publication identity. No model execution or SDK publication occurred.

`publish_hf_docs.py` defaults to read-only preflight and refuses a changed parent or
candidate. It is a bounded publisher for this recorded transition, not a general
release command; it will refuse to republish now that HF main has advanced.
