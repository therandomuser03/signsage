# Credits and Licenses

SignSage stands on other people's work. This file is the single source for attribution; the in-app Credits page should mirror it ([DESIGN_SYSTEM §8](docs/DESIGN_SYSTEM.md#8-credits-and-attribution-required)).

## The people

**SignSage V1 (ASL ↔ English)** — the prototype this project extends:

- [Anwesa Sinha](https://www.linkedin.com/in/anwesa-sinha/)
- [Draksha](https://www.linkedin.com/in/draksha-aa3164227/)
- [Satyaki Dawn](https://www.linkedin.com/in/satyaki-dawn-116420230/)
- [Subhadip Das](https://www.linkedin.com/in/subhadip-das-7a150319b/)

The current ISL version is a re-implementation and research extension of that work, developed as an M.Tech CSE project at Vidyavardhaka College of Engineering (VVCE), Mysuru. Attribution and license terms of the original components remain in each subdirectory's own LICENSE files.

## Datasets and research

| Resource | Credit | License / terms |
|---|---|---|
| **FDMSE-ISL** (isolated ISL words) | Patra, Maitra, Tiwari, Kumaran, Prabhu, Punyeshwarananda, Samanta — portal `cs.rkmvu.ac.in/~isl/isl-dataset/` | Used with the authors' permission for this project. Research use; **do not redistribute**. Confirm permission for a public app in writing |
| iSign v1.1 | Joshi et al., *iSign*, ACL Findings 2024 (`Exploration-Lab/iSign`) | CC BY-NC-SA 4.0 (gated); derived weights do not inherit MIT |
| INCLUDE and the `isl-isolated-40words` aggregate | Their authors; aggregate by `vidit031` on Hugging Face | INCLUDE subset used for the early pilot recorded as CC-BY-4.0; the aggregate is multi-source (`other`) — check each source |
| RealSign Indian Sign Language dataset | Its authors | CC0-1.0 |
| Indian Sign Language Real-life Words (Mendeley) | Its authors | Check the source |
| ISLRTC dictionary videos and learning materials | Indian Sign Language Research and Training Centre | Reference material; check terms before redistributing |

**Please cite FDMSE-ISL** (BibTeX in [docs/references.bib](docs/references.bib)):

> Patra, S., Maitra, A., Tiwari, M., Kumaran, K., Prabhu, S., Punyeshwarananda, S., Samanta, S. *Hierarchical Windowed Graph Attention Transformer Encoder and a Large Scale Dataset for Indian Sign Language Recognition.* Pattern Analysis and Applications 28(3):148, 2025. doi:10.1007/S10044-025-01529-3

Models evaluated and not shipped (credit for the research, no endorsement implied): ISLiM, Sanket (MIT), Sid-V5 (MIT), `t5-small-motion` (`manavdhamecha77/iSign-t5-pose-to-text`).

## Third-party software

Full dependency lists: `backend/pyproject.toml`, `backend/requirements.txt`, `frontend/package.json`. Components that need a specific notice:

### Piper TTS (voice engine) — GPL-3.0-or-later
[piper-tts](https://github.com/OHF-Voice/piper1-gpl) is imported lazily by `backend/app/services/tts_service.py` (the packaged `piper` module). It is licensed under the **GNU General Public License v3.0 or later**; see the upstream repository for the license text. Review how it is combined with the MIT-licensed backend before distributing a build that includes it.

### Piper voice `en_US-kristin-medium`
Default server-side voice (a free, offline alternative to a paid TTS API). US English, single speaker, medium quality, 22,050 Hz. Files `en_US-kristin-medium.onnx` and `.onnx.json` are downloaded from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) at build time or into `SIGNSAGE_PIPER_CACHE`; they are not committed. Trained on LibriVox recordings (≈ 11.5 hours); the voice and data are **public domain** per the model card. Chosen because its license places no restriction on this use.

### Groq Orpheus (optional)
`canopylabs/orpheus-v1-english` is used only when the Groq provider is selected. It needs a Groq API key and a one-time acceptance of the model terms in the Groq console by the key's owner.

### Other
MediaPipe (Apache-2.0), FastAPI, PyTorch, React, Vite, Tailwind CSS, shadcn/ui, Three.js, Supabase, Vercel and Render are used under their own licenses. The avatar uses a MakeHuman-style rig (`signsage-human.glb`) — record the asset's license here.

## Licensing at a glance

| Thing | License |
|---|---|
| SignSage source code | MIT ([LICENSE](LICENSE)) |
| Datasets | Their own terms (table above); never committed |
| Model weights | Follow the training data's terms; **do not inherit MIT** |
| Avatar motion clips | Derived from the recordings they came from; check each source before shipping |

## Adding a credit

When you add a dataset, model, asset or library with a notice requirement: add a row here (name, authors, license, how it is used), update the in-app Credits page, and mention it in [CHANGELOG.md](CHANGELOG.md).
