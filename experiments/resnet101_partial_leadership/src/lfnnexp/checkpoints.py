from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def extract_state_dict(checkpoint: Any, key: str = "auto") -> dict[str, torch.Tensor]:
    if key != "auto":
        if not isinstance(checkpoint, dict) or key not in checkpoint:
            raise KeyError(f"Checkpoint has no state-dict key {key!r}.")
        state = checkpoint[key]
    elif isinstance(checkpoint, dict):
        state = checkpoint
        for candidate in ("model", "state_dict", "model_state_dict"):
            if candidate in checkpoint and isinstance(checkpoint[candidate], dict):
                state = checkpoint[candidate]
                break
    else:
        raise TypeError("Checkpoint must contain a state dictionary.")
    if not isinstance(state, dict) or not state:
        raise TypeError("Resolved state dictionary is empty or invalid.")
    cleaned: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        new_name = str(name)
        while new_name.startswith("module."):
            new_name = new_name[len("module.") :]
        cleaned[new_name] = value
    return cleaned

