"""English public playground for the released Qwen3.5 classification checkpoint."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

ZERO_GPU = os.environ.get("QEV_ZERO_GPU") == "1"
if ZERO_GPU:
    import spaces

# ZeroGPU must patch CUDA before the runtime imports PyTorch.
import gradio as gr  # noqa: E402
from engine import DemoEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent
PRESETS = json.loads((ROOT / "presets.json").read_text("utf-8"))
LABELS = {
    "support": "Text / Route a support request",
    "policy": "Text / Apply an order policy",
    "uncertain": "Text / Check insufficient evidence",
    "photo": "Image / Choose the material",
    "photo_score": "Image / Score visibility",
    "photo_truth": "Image / Judge a proposition",
    "photo_policy": "Image + text / Apply a sorting policy",
}
SAMPLES = [
    ("sample-01.jpg", "Glass bottle"),
    ("sample-02.jpg", "Newspaper"),
    ("sample-03.jpg", "Cardboard"),
    ("sample-04.jpg", "Plastic bottle"),
    ("sample-05.jpg", "Metal can"),
    ("sample-06.jpg", "Packaging pouch"),
]
CSS = """
.gradio-container { max-width: 1200px !important; }
#intro { padding: 12px 0 20px; }
#intro h1 { letter-spacing: -.04em; font-size: 34px; line-height: 1.12; }
#intro p { max-width: 760px; }
#run-button { min-height: 48px; }
footer { display: none !important; }
"""


def preset_values(key):
    preset = PRESETS[key]
    question = preset["questions"][0]
    options = question["options"]
    criteria = "\n".join(
        item["text"] if question["type"] == "score" else f"{item['key']} | {item['text']}"
        for item in options
    )
    image = str(ROOT / "samples" / (preset["sampleId"] + ".jpg")) if "sampleId" in preset else None
    return preset["text"], image, question["instructions"], question["type"], criteria


def default_criteria(kind):
    if kind == "noul":
        return "false | The proposition is false.\ntrue | The proposition is true."
    if kind == "score":
        return (
            "Low: the criterion is not met.\nMedium: the criterion is partly met.\n"
            "High: the criterion is fully met."
        )
    return "yes | The evidence meets the criterion.\nno | The evidence does not meet the criterion."


def runtime_notice(device, shared_gpu=False):
    if shared_gpu:
        return (
            "**Shared GPU demo.** GPU acceleration is enabled. Response time depends on "
            "GPU availability, input size and the queue. Daily usage limits apply."
        )
    if device == "cuda":
        return (
            "**GPU demo.** GPU acceleration is enabled. "
            "Response time depends on input size and the queue."
        )
    return (
        "**CPU demo.** Responses may take several seconds. GPU hosting can make model "
        "processing faster; input size and queue time also affect how long you wait."
    )


def build_demo(engine):
    def predict(*args):
        try:
            return engine.predict(*args)
        except (ValueError, TypeError) as error:
            raise gr.Error(str(error)) from error
        except Exception as error:
            logging.exception("Demo inference failed")
            raise gr.Error(
                "The model could not process this request. Please try a shorter example."
            ) from error

    if ZERO_GPU:
        predict = spaces.GPU(duration=30)(predict)

    initial = preset_values("support")
    with gr.Blocks(title="QEV", delete_cache=(300, 600)) as demo:
        gr.Markdown(
            "# QEV\n"
            "Provide the evidence. Define your candidates. Inspect the probabilities.\n\n"
            "Dynamic text and image decisions with **Qwen3.5-2B**. "
            "[Model](https://huggingface.co/ken-jo/qev) · "
            "[Data](https://huggingface.co/datasets/ken-jo/qev-data) · "
            "[Source](https://github.com/ken-jo/qev)",
            elem_id="intro",
        )
        gr.Markdown(runtime_notice(engine.device, shared_gpu=ZERO_GPU))
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                example = gr.Dropdown(
                    choices=[(value, key) for key, value in LABELS.items()],
                    value="support",
                    label="Start with an example",
                )
                evidence = gr.Textbox(
                    value=initial[0], label="Evidence and context", lines=4, max_lines=8
                )
                image = gr.Image(
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    label="Image (optional)",
                    height=250,
                )
                gr.Examples(
                    examples=[[str(ROOT / "samples" / name)] for name, _ in SAMPLES],
                    inputs=[image],
                    label="Sample photographs",
                    cache_examples=False,
                )
                gr.Markdown(
                    "Photos: TrashNet, Gary Thung & Mindy Yang (MIT). These examples overlap "
                    "development data; they are not a benchmark."
                )
                kind = gr.Radio(
                    ["choice", "score", "noul"], value=initial[3], label="Decision type"
                )
                question = gr.Textbox(value=initial[2], label="Question and decision rule", lines=2)
                criteria = gr.Textbox(
                    value=initial[4], label="Candidate descriptions / score levels", lines=5
                )
                gr.Markdown(
                    "**choice:** `label | description`, one per line. "
                    "**score:** one description per line, from level 0 upward. "
                    "**noul:** use `false | ...` and `true | ...`."
                )
                resolution = gr.Dropdown(
                    ["Original", "64", "128", "192", "256", "384", "512", "768", "1024"],
                    value="Original",
                    label="Image long edge (pixels)",
                    info=(
                        "Resize without cropping or upscaling. "
                        "The model then applies its trained pixel budget."
                    ),
                )
                run = gr.Button("Run decision", variant="primary", elem_id="run-button")
            with gr.Column(scale=5):
                summary = gr.Textbox(label="Decision", lines=6, interactive=False)
                distribution = gr.Label(label="Candidate probabilities", num_top_classes=16)
                gr.Markdown(
                    "Probabilities are model estimates, not verified accuracy. "
                    "An abstained prediction needs review. "
                    "Score is an expected level, not a confidence percentage."
                )
                with gr.Accordion("Request and response JSON", open=False):
                    request = gr.Code(label="Request", language="json", interactive=False)
                    response = gr.JSON(label="Response")
                with gr.Accordion("Scope and limitations", open=False):
                    gr.Markdown(
                        "One image, one question and 2–16 candidates per demo request. "
                        "The full API supports up to four questions. OCR and spatial reasoning "
                        "are weak; 2048 experiments produced no wins. English is the main "
                        "evaluated language. Uploaded inputs are processed on this Space; "
                        "temporary upload files expire after approximately 10 minutes. "
                        "Avoid uploading confidential material."
                    )

        def clear_results():
            return "Inputs changed. Run a decision to see the new result.", None, None, None

        result_outputs = [summary, distribution, request, response]
        example.change(
            preset_values,
            [example],
            [evidence, image, question, kind, criteria],
            queue=False,
            api_name=False,
        ).then(clear_results, [], result_outputs, queue=False, api_name=False)
        kind.input(default_criteria, [kind], [criteria], queue=False, api_name=False)
        for control in (evidence, image, question, kind, criteria, resolution):
            control.input(clear_results, [], result_outputs, queue=False, api_name=False)
        run.click(
            predict,
            [evidence, image, question, kind, criteria, resolution],
            [summary, distribution, request, response],
            api_name="predict",
            concurrency_limit=1,
            concurrency_id="model",
        )
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = DemoEngine(
        device="cuda" if ZERO_GPU else os.environ.get("QEV_DEVICE", "cpu"),
        checkpoint=os.environ.get("QEV_CHECKPOINT_PATH"),
    ).load()
    demo = build_demo(engine)
    demo.queue(max_size=8, default_concurrency_limit=1)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=False,
        max_file_size="10mb",
        css=CSS,
        theme=gr.themes.Base(primary_hue="blue"),
    )
