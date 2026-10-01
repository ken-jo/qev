"""Qwen3.5 Classification public facade over the byte-preserved evaluated Veyra runtime."""

from veyra.option_model import OptionModel as QwenClassification
from veyra.schema import DecisionRequest

__version__ = "0.1.0"
__all__ = ["DecisionRequest", "QwenClassification", "__version__"]
