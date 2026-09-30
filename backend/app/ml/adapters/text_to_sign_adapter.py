"""Text -> Sign adapter (neural pose generation).

Status: NOT IMPLEMENTED / NOT SERVED. See ml/models/text_to_sign.py. The
shipped English->ISL direction is the VRM avatar + gloss-lookup pipeline --
see api/routes/avatar_gloss.py and services/avatar_gloss_service.py -- which
deliberately does not go through this ml/ layer at all, because it needs no
trained model. This file exists only so the folder structure documents
where a future generative Text->Pose adapter would live if that direction
is revived and re-evaluated.
"""
from __future__ import annotations


class TextToSignAdapter:
    def __init__(self, *_, **__):
        raise NotImplementedError(
            "Not served. Use api/routes/avatar_gloss.py (rule-based gloss + "
            "pre-authored avatar animation) for English->ISL instead."
        )
