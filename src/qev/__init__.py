"""QEV public interface to the evaluated multimodal decision runtime."""

__version__ = "0.1.1"
__all__ = ["DecisionRequest", "QEV", "__version__"]


def __getattr__(name):
    # CLI help and packaging metadata do not need to initialize PyTorch.
    if name == "QEV":
        from veyra.option_model import OptionModel

        return OptionModel
    if name == "DecisionRequest":
        from veyra.schema import DecisionRequest

        return DecisionRequest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
