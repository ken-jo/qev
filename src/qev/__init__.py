"""QEV public interface to the evaluated multimodal decision runtime."""

__version__ = "0.2.0"
__all__ = ["DecisionRequest", "QEV", "load", "__version__"]


def load(checkpoint=None, *, device="auto", cache_dir=None, offline=False):
    """Load QEV, downloading its pinned weights on the first use.

    Later calls reuse the local model cache. ``offline=True`` requires all weights
    to be available already. ``QEV.load`` remains the original low-level loader.
    """
    from qev.runtime import load_model

    return load_model(checkpoint, device=device, cache_dir=cache_dir, offline=offline)


def __getattr__(name):
    # CLI help and packaging metadata do not need to initialize PyTorch.
    if name == "QEV":
        from veyra.option_model import OptionModel

        return OptionModel
    if name == "DecisionRequest":
        from veyra.schema import DecisionRequest

        return DecisionRequest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
