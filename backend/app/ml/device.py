"""Single place that turns a configured device string into a torch.device.

"auto" (and "" / None) means cuda when available, else cpu. cpu, cuda,
cuda:N and mps pass through unchanged. Anything else falls back to cpu with
a warning instead of crashing torch.device().
"""
from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger(__name__)

_VALID = re.compile(r"(cpu|cuda|cuda:[0-9]+|mps)")


def resolve_torch_device(value: object) -> torch.device:
    text = "" if value is None else str(value).strip().lower()
    if text in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _VALID.fullmatch(text):
        return torch.device(text)
    logger.warning("Unrecognised device %r; falling back to cpu", value)
    return torch.device("cpu")
