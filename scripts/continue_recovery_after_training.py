"""Continue this authorized experiment after its exact training process exits successfully."""

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(pid):
    root = Path("runs/backbone-recovery-chain-v13")
    root.mkdir(parents=True, exist_ok=False)
    progress = {"stage": "training", "status": "waiting", "training_process_id": pid}

    def record(**fields):
        progress.update(fields)
        progress.update(
            updated_at_utc=datetime.now(timezone.utc).isoformat(), release_allowed=False
        )
        (root / "progress.json").write_text(
            json.dumps(progress, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(progress), flush=True)

    try:
        record(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.QueryFullProcessImageNameW.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x00101000, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            buffer, length = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
            if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
                raise ctypes.WinError(ctypes.get_last_error())
            if Path(buffer.value).resolve() != Path(sys.executable).resolve():
                raise ValueError("Expected the already running project Python process")
            while True:
                status = kernel.WaitForSingleObject(handle, 30000)
                if status == 0:
                    break
                if status != 258:
                    raise ctypes.WinError(ctypes.get_last_error())
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if exit_code.value != 0:
                raise RuntimeError(f"Training process failed with exit code {exit_code.value}")
        finally:
            kernel.CloseHandle(handle)
        study = Path("runs/backbone-recovery-memory-v13")
        if read(study / "progress.json").get("stage") != "complete":
            raise ValueError("The exact training and merged development study did not complete")
        if read(study / "selection.json").get("eligible") is not True:
            raise ValueError("No candidate passed the fixed development requirements")
        stages = (
            ("admission", ["scripts/prepare_backbone_recovery_policy.py"]),
            ("policy-validation", ["scripts/run_recovery_policy_validation.py"]),
            ("final-evaluation", ["scripts/run_recovery_final_release.py", "--evaluate-only"]),
        )
        for stage, arguments in stages:
            record(stage=stage, status="running")
            with (root / (stage + ".log")).open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    [sys.executable, "-u", *arguments],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if completed.returncode:
                raise RuntimeError(f"{stage} failed with exit code {completed.returncode}")
            record(status="complete")
        record(
            stage="acceptance-complete",
            status="complete",
            remaining="Update result documentation, prepare HF and verify bundled-wheel inference",
        )
    except Exception as error:
        record(status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-pid", type=int, required=True)
    main(parser.parse_args().training_pid)
