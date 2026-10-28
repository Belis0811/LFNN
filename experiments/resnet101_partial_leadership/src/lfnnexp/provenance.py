from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *command],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_state(cwd: str | Path) -> dict[str, Any]:
    root = Path(cwd).resolve()
    top_level = _git(["rev-parse", "--show-toplevel"], root)
    if top_level is None:
        return {"is_git_repository": False}
    git_root = Path(top_level)
    status = _git(["status", "--porcelain"], git_root)
    return {
        "is_git_repository": True,
        "root": str(git_root),
        "commit": _git(["rev-parse", "HEAD"], git_root),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], git_root),
        "dirty": bool(status),
        "dirty_entry_count": 0 if not status else len(status.splitlines()),
    }


def assert_clean_git(cwd: str | Path, allow_dirty: bool) -> dict[str, Any]:
    state = git_state(cwd)
    if not state.get("is_git_repository"):
        raise RuntimeError(
            "The experiment source is not inside a Git repository. Initialize and "
            "commit the source before a reported run."
        )
    if state.get("dirty") and not allow_dirty:
        raise RuntimeError(
            "The Git worktree is dirty. Commit the exact source/config first, or use "
            "--allow-dirty only for a nonreported test."
        )
    return state


def collect_metadata(
    *,
    config: dict[str, Any],
    source_files: Iterable[str | Path],
    git: dict[str, Any],
) -> dict[str, Any]:
    try:
        import torch
        import torchvision

        torch_info: dict[str, Any] = {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as error:  # metadata collection must not mask the real issue
        torch_info = {"collection_error": repr(error)}

    hashes = {}
    for item in source_files:
        path = Path(item).resolve()
        if path.is_file():
            hashes[str(path)] = sha256_file(path)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "working_directory": str(Path.cwd().resolve()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "process_id": os.getpid(),
        "git": git,
        "software": torch_info,
        "source_sha256": hashes,
        "resolved_config": {
            k: v for k, v in config.items() if not k.startswith("_")
        },
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.handle = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, value: dict[str, Any]) -> None:
        self.handle.write(json.dumps(value, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

