# Ordered multi-image input

This is an unreleased source extension. Published SDK 0.2.1 accepts one image.
The source SDK, HTTP API and playground accept up to **four ordered images**.
The learned QEV checkpoint and Qwen revision are unchanged. Use the development checkout
in its own environment; do not replace an environment used by an ongoing training run.

## Compare a reference and current state

```json
{
  "state": {
    "text": "Image 1 is the reference. Image 2 is the current state.",
    "images": [{"path": "reference.png"}, {"path": "current.png"}]
  },
  "questions": {
    "same_state": {
      "type": "noul",
      "instructions": "Image 2 shows the same equipment state as Image 1.",
      "criteria": {
        "true": "The visible equipment state matches the reference.",
        "false": "The visible equipment state differs from the reference."
      }
    }
  }
}
```

Put both files under the configured image root. Image order is retained. Multi-image
prompts insert `Image 1:`, `Image 2:`, etc. before separate visual placeholders;
no collage is generated. State text describes image roles, and the question defines
what should match. The [choice example](../examples/multi_image/request.json) also
provides a review candidate for obscured or insufficient evidence. Its filenames are
placeholders for user-supplied files, and it contains no measured prediction.

## Limits and interface

- Maximum four images and four questions; 2-16 candidates per choice/score.
- Existing per-image preprocessing budgets remain 65,536-262,144 pixels by default.
- Each source file must be local, at most 20 MiB and at most 40 megapixels before decode.
  Every image is checked against the configured root, including resolved symlinks.
- The default 2,048 processed-token budget is for the complete request, including
  image tokens, questions, candidate text and decision views. Excess input fails before
  the backbone forward; images and text are not silently omitted or truncated.
- Playground: fill Image 1, then the slots under **Additional images**, in order.
  The selected long-edge setting applies to every image. Upload limits remain 10 MB
  and 16 megapixels per image. Selecting a preset clears additional image slots.
- `usage.input_images` reports the number of images. Output types and candidate
  probability semantics remain the same.

## Evaluation scope

Code supports multiple visual inputs, but runtime correctness, relative-image reasoning,
accuracy, calibration and latency have not yet been measured for this extension.
The existing calibration and abstention parameters were fitted on the historical
population; applying them does not establish calibrated multi-image confidence.
Published single-image accuracy and timing do not establish multi-image performance.
QEV's qualification target is **100 ms or less** on the declared timing workload.
Vega's separate 150 ms requirement does not apply to QEV. Multi-image timing requires
a separate measured workload.
