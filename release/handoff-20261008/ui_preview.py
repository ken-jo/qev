"""Render the packaged playground for UI inspection without loading model weights."""

import gradio as gr

from qev.playground.app import CSS, ROOT, build_demo, english_ui_i18n


class UIOnlyEngine:
    device = "cpu"

    def predict(self, *args):
        raise RuntimeError("Inference is disabled in this UI-only preview.")


if __name__ == "__main__":
    demo = build_demo(UIOnlyEngine())
    demo.queue(max_size=8, default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7863,
        share=False,
        inbrowser=False,
        show_error=False,
        max_file_size="10mb",
        css=CSS,
        theme=gr.themes.Base(primary_hue="blue"),
        i18n=english_ui_i18n(),
        allowed_paths=[str(ROOT / "samples")],
    )
