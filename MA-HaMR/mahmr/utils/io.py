from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, obj: Dict[str, Any], indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def save_torch(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path)


def load_torch(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)
