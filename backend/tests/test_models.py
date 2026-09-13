from pathlib import Path

import numpy as np
import torch

from app.models.sign_to_text import SignTransformer, TextToPoseGRU
from app.services.vocab import Vocabulary


ROOT = Path(__file__).parents[1]


def _load(path, model, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()


def test_transformer_checkpoint_loads():
    vocab = Vocabulary.from_json(ROOT / "vocab" / "token_vocabulary.json")
    model = SignTransformer(1629, vocab.size, vocab.pad_id)
    _load(ROOT / "weights" / "signsage_transformer_full.pt", model)


def test_noface_checkpoint_loads():
    vocab = Vocabulary.from_json(ROOT / "vocab" / "token_vocabulary.json")
    model = SignTransformer(225, vocab.size, vocab.pad_id)
    _load(ROOT / "weights" / "signsage_transformer_noface.pt", model)


def test_text_to_pose_checkpoint_loads():
    vocab = Vocabulary.from_json(ROOT / "vocab" / "gloss_vocabulary.json")
    model = TextToPoseGRU(vocab.size, 1629, vocab.pad_id)
    _load(ROOT / "weights" / "signsage_text_to_pose_best.pt", model)


def test_transformer_cpu_inference_smoke():
    vocab = Vocabulary.from_json(ROOT / "vocab" / "token_vocabulary.json")
    model = SignTransformer(1629, vocab.size, vocab.pad_id)
    _load(ROOT / "weights" / "signsage_transformer_full.pt", model)

    frames = torch.zeros((1, 8, 1629), dtype=torch.float32)
    lengths = torch.tensor([8])
    idx = torch.arange(8).unsqueeze(0)
    pad = idx >= lengths.unsqueeze(1)
    memory = model.encode(frames, pad)

    ids = torch.tensor([[vocab.bos_id]])
    mask = torch.triu(torch.ones(1, 1, dtype=torch.bool), diagonal=1)
    logits = model.decode(ids, memory, mask, pad)
    assert logits.shape == (1, 1, vocab.size)


def test_text_to_pose_cpu_inference_smoke():
    vocab = Vocabulary.from_json(ROOT / "vocab" / "gloss_vocabulary.json")
    model = TextToPoseGRU(vocab.size, 1629, vocab.pad_id)
    _load(ROOT / "weights" / "signsage_text_to_pose_best.pt", model)

    tokens = torch.tensor([[vocab.bos_id, vocab.eos_id]])
    with torch.inference_mode():
        pose = model(tokens)
    assert pose.shape == (1, 60, 1629)
    assert np.isfinite(pose.numpy()).all()
