"""Optional cosmetic text refinement.

This is NOT part of the trained model or the notebook's evaluation --
purely a presentation-layer touch-up (capitalize first letter, ensure
terminal punctuation) applied AFTER acceptance. Never call this before the
confidence/motion gates in confidence.py, and never let it change whether a
result is accepted -- it only affects display formatting of already-
accepted text.
"""
from __future__ import annotations


def refine_for_display(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text
