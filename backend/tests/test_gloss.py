from app.services.gloss import naive_gloss


def test_naive_gloss_is_deterministic():
    assert naive_gloss("This is a test for the system") == "test system"


def test_naive_gloss_maps_unseen_greeting_to_vocabulary_word():
    assert naive_gloss("Hi") == "hello"
