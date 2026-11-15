"""Unit tests for app.ml.device.resolve_torch_device."""
from __future__ import annotations

import logging

import torch

from app.ml.device import resolve_torch_device


def test_auto_gives_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_torch_device("auto") == torch.device("cpu")
    assert resolve_torch_device("") == torch.device("cpu")
    assert resolve_torch_device(None) == torch.device("cpu")


def test_auto_gives_cuda_when_cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_torch_device("auto") == torch.device("cuda")


def test_cpu_stays_cpu():
    assert resolve_torch_device("cpu") == torch.device("cpu")


def test_valid_values_pass_through():
    assert resolve_torch_device("cuda") == torch.device("cuda")
    assert resolve_torch_device("cuda:0") == torch.device("cuda:0")
    assert resolve_torch_device("mps") == torch.device("mps")


def test_invalid_string_gives_cpu_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="app.ml.device"):
        assert resolve_torch_device("tpu") == torch.device("cpu")
    assert "Unrecognised device" in caplog.text
