"""Dispatch only explicitly supported continuation studies to their evidence auditors."""

import hashlib
import json
from pathlib import Path


def collect_release_extension_evidence(
    checkpoint, selection_path, records, calibration_report, policy_report
):
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    extensions = set(selection.get("release_extensions", {}))
    if extensions == {"workflow_workspace_v12"}:
        from workspace_release_checks import collect_workspace_evidence as collect
    elif extensions == {"workflow_curriculum_v12"}:
        from curriculum_release_checks import collect_curriculum_evidence as collect
    elif extensions == {"workflow_depth_v12"}:
        from depth_release_checks import collect_depth_evidence as collect
    elif extensions == {"workflow_depth_policy_v12"}:
        from depth_policy_checks import collect_policy_evidence as collect
    elif extensions == {"workflow_cohort_policy_v12"}:
        from cohort_policy_checks import collect_cohort_evidence as collect
    else:
        raise ValueError("unsupported or combined release-study extensions")
    result = collect(checkpoint, selection_path, records, calibration_report, policy_report)
    path = Path("scripts/workflow_extension_checks.py")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    result["source_files"][str(path)] = checksum
    result["evidence_files"][str(path)] = checksum
    return result
