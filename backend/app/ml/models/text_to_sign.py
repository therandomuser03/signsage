"""Text -> Sign neural pose generator.

Status: NOT SERVED. The training notebook's Text->Pose GRU experiment is
gated behind RUN_TEXT_TO_POSE and is not executed by the current notebook
run (see docs/UPDATE_NOTES.md: "Latest notebook does not run Text->Pose;
the 20260916 result is historical only", and RESULTS.md: generated motion
was only 7.9% of ground-truth motion -- near-static pose collapse).

The actual shipped Text->Sign direction is the pre-authored VRM avatar +
gloss-lookup pipeline in frontend/src/features/text-to-sign/ and backend
routes/avatar_gloss.py -- NOT a generative model. See
ml/adapters/text_to_sign_adapter.py.

This module is a placeholder so ml/models/ mirrors the notebook's full
scope without pretending an unvalidated model is production code. Do not
implement this by guessing at the GRU architecture from memory; port it
verbatim from the notebook's own Section 12 cells if/when that direction
is revived and re-evaluated.
"""
from __future__ import annotations


class TextToPoseModel:
    def __init__(self, *_, **__):
        raise NotImplementedError(
            "Text->Pose generative model is not served (near-static pose collapse, "
            "see RESULTS.md). The shipped English->ISL path is the VRM avatar/gloss "
            "system -- see ml/adapters/text_to_sign_adapter.py."
        )
