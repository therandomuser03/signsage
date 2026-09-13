from pathlib import Path

from app.services.vocab import Vocabulary


ROOT = Path(__file__).parents[1]


def test_token_vocab_loads():
    vocab = Vocabulary.from_json(ROOT / "vocab" / "token_vocabulary.json")
    assert vocab.size > 4
    assert vocab.encode("hello world", 40)[0] == vocab.bos_id


def test_gloss_vocab_loads():
    vocab = Vocabulary.from_json(ROOT / "vocab" / "gloss_vocabulary.json")
    assert vocab.size > 4
    assert vocab.encode("hello world", 40)[0] == vocab.bos_id
