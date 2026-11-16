# Third-party notices

SignSage ships and uses the following third-party components. This file records
the ones requiring notice; the full dependency list lives in `backend/pyproject.toml`
and `backend/requirements.txt`.

## Piper TTS (voice engine) — GPL-3.0-or-later

[piper-tts](https://github.com/OHF-Voice/piper1-gpl) (imported lazily by
`backend/app/services/tts_service.py` as the packaged `piper` module) is
licensed under the **GNU General Public License v3.0 or later**. It runs as a
separate process component within the backend image; the SignSage backend
communicates with it in-process. See the upstream repository for the license text.

## Piper voice: en_US-kristin-medium

Used as the default server-side text-to-speech voice (a free, offline
alternative to a paid TTS API).

* **Voice name:** `en_US-kristin-medium` (US English female, single speaker,
  medium quality, 22,050 Hz)
* **Files:** `en_US-kristin-medium.onnx` + `en_US-kristin-medium.onnx.json`,
  downloaded at Docker build time from the public
  [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) repository
  (paths `en/en_US/kristin/medium/`), or at runtime into `SIGNSAGE_PIPER_CACHE`.
  The files are not committed to this repository.
* **Dataset:** LibriVox (https://librivox.org), ~11.5 hours of recordings
* **License of the voice/dataset:** **public domain** (per the model card and
  the author's statement; LibriVox content is public domain in the US)
* **Model card:** https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/kristin/medium/MODEL_CARD

Chosen because its license places no restriction on this use (unlike
CC BY-NC-SA voices, which forbid commercial use).

## Groq Orpheus (optional, not shipped by default)

`canopylabs/orpheus-v1-english` is used only when `SIGNSAGE_TTS_PROVIDER=groq`
(or the user picks "Groq Orpheus" in the UI). Access requires a Groq API key
and a one-time acceptance of the model terms in the Groq console, per the
organisation that owns the key. It is not used in `auto` mode.
