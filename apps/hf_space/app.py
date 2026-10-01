"""Compatibility launcher; the English playground is included in the qev package."""

import os

from qev.playground.app import launch

if __name__ == "__main__":
    launch(
        checkpoint=os.environ.get("QEV_CHECKPOINT_PATH"),
        device=os.environ.get("QEV_DEVICE", "auto"),
        cache_dir=os.environ.get("QEV_CACHE_DIR"),
        offline=os.environ.get("QEV_OFFLINE") == "1",
        host=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        port=int(os.environ.get("PORT", "7860")),
    )
