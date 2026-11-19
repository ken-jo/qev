"""Prompt markers of the QEV decision prompt.

`TRAINED_RESULT_MARKER` is the trained read position of every published checkpoint: the
decision state is read at its final ":". Changing its bytes changes model outputs.
The packed-prompt markers are used only by `packing.pack_request`, which the published
option-readout checkpoints do not use.
"""

# Packed-prompt markers (`packing.pack_request`).
TRAINED_CANDIDATE_MARKER = "[QEV_CANDIDATE]"
TRAINED_QUESTION_MARKER = "[QEV_QUESTION]"
# Trained read position of every published checkpoint. Do not change these bytes.
TRAINED_RESULT_MARKER = "VeyraResult:"
