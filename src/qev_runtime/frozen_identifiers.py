"""Identifiers that keep the project's former name because published artifacts need them.

Each value is a model input that the published weights were trained on, a salt that fixes
recorded record IDs and splits, or a name inside artifacts published before the rename.
Renaming any of them changes model outputs or stops published artifacts from being read.
Everything new writes QEV names; these constants are only read or reproduced.
"""

# Trained prompt markers: inputs of every published checkpoint. Do not rename.
TRAINED_CANDIDATE_MARKER = "[VEYRA_CANDIDATE]"
TRAINED_QUESTION_MARKER = "[VEYRA_QUESTION]"
TRAINED_RESULT_MARKER = "VeyraResult:"

# Record-ID salts of the published Foundation v11, Workflow v12 and workspace-calibration
# corpora. Changing them changes every record ID and split assignment.
FOUNDATION_V11_ID_SALT = "veyra-foundation-v11-83"
WORKFLOW_V12_ID_SALT = "veyra-workflow-v12-131"
WORKSPACE_CALIBRATION_ID_SALT = "veyra-workspace-calibration-157"

# Loader file name inside Hugging Face bundles published before the rename (read only).
LEGACY_BUNDLE_LOADER = "load_veyra.py"

# Request header sent by playground clients published before the rename (read only).
LEGACY_PLAYGROUND_HEADER = "X-Veyra-Playground"
