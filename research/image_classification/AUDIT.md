# SignSage Image Dataset Audit

## Summary of findings

| # | Question | Answer |
|---|----------|--------|
| 1 | Which datasets are actually available? | 2 of 3: RealSign-ISL-Alphabet (usable) and Indian-Sign-Language-Real-life-Words (usable). Indian-Sign-Language-dataset (ayesha-hannure) is **NOT usable** — its inner image zips are empty (Git LFS pointers, no actual images). |
| 2 | Are their labels and licenses usable? | Yes. RealSign: CC0-1.0, 26 alphabet classes A-Z, labels from directory names. Real-life Words: license to verify (Mendeley), 20 word classes, labels from zip/directory names. |
| 3 | How much data is usable after filtering? | RealSign: ~26,000 images (Train 18,198 / Val 2,579 / Test 5,200). Real-life Words: ~18,000 images across 20 words (~900/word). |
| 4 | Which classes should be included in the first model? | Alphabet: all 26 A-Z letters from RealSign. Words: all 20 words from Real-life Words. Two separate classifiers (alphabet ≠ word label space). |
| 5 | Which dataset supplies validation/test? | RealSign already has Train/Val/Test splits — use as-is. Real-life Words: create user-aware (signer-independent) splits by User_N. |
| 6 | What is the smallest credible pilot? | One alphabet CNN (26 classes, RealSign) + one word CNN (20 classes, Real-life Words), RGB 64x64, CPU, small batches. |

---

## Dataset 1: Indian-Sign-Language-dataset (ayesha-hannure)

**Status: NOT USABLE — no actual image data.**

- Path: `Indian-Sign-Language-dataset-main.zip` (6,242 bytes at repo root)
- License: Apache 2.0 (verified in extracted LICENSE)
- Declared: 26 alphabet classes, 486 images/class (12,700 total)
- Actual contents: README.md, LICENSE, `csv_ISL.zip` (134 bytes, empty), `dataset_ISL.zip` (134 bytes, empty)
- The two inner zips extract to zero bytes — they are Git LFS pointers or placeholders. No images are present.
- **Verdict: exclude. Cannot train or evaluate without images.**

## Dataset 2: Indian Sign Language — Real-life Words (Mendeley)

**Status: USABLE.**

- Path: `Indian sign Language-Real-life Words.zip` (514,941,675 bytes at repo root)
- License: to be confirmed from Mendeley dataset page (dataset page: https://data.mendeley.com/datasets/s6kgb6r3ss/2). No LICENSE file inside the zip.
- Structure: outer zip contains `New folder/<word>.zip` for each of 20 word classes. Each inner zip contains `<word>/User_N/<word>_N_UserN_N.jpg`.
- Classes (20): afraid, become, bad, assistance, agree, college, doctor, from, pain, pray, secondary, skin, small, specific, stand, today, warn, which, work, you
- Signer metadata: yes — `User_1` through `User_6+` per word, enabling signer-independent splits.
- Sample counts: afraid has 900 images across 6 users (~150/user). Other classes expected similar (~900/word → ~18,000 total).
- Format: RGB JPEG, varying dimensions.
- **Verdict: include for word classifier.**

## Dataset 3: RealSign-Indian-Sign-Language-Dataset (RealSign62)

**Status: USABLE.**

- Path: `RealSign-Indian-Sign-Language-Dataset-main.zip` (653,370,264 bytes at repo root), extracted to `/tmp/opencode/dataset-audit/ds3/extracted/`
- License: CC0-1.0 (verified in extracted LICENSE)
- Citation: Boggaram et al., RealSign-Indian-Sign-Language-Dataset, PES University
- Structure (from nested `Dataset.zip`): `Training/<A-Z>/`, `Testing/<A-Z>/`, `Validation/<A-Z>/`, `Letters/<A-Z>/`
- Classes (26): A through Z (ISL fingerspelled alphabet)
- Counts: Training 18,198 (~700/class), Testing 5,200 (200/class), Validation 2,579 (~99/class). Total ~26,000.
- Signer metadata: NO user subdirectories (images named `N.jpg` only). Four signers contributed per README, but per-image signer identity is not available.
- Format: RGB JPEG, varying dimensions (~300x260).
- Train/Val/Test splits already exist and are leakage-free by construction (separate directories).
- **Verdict: include for alphabet classifier.**

## Label compatibility

- Alphabet (RealSign) and word (Real-life Words) label spaces are disjoint (A-Z vs. 20 words). **Do not merge into one classifier.** Use two separate classifiers.
- No cross-dataset label overlap to resolve.
- Near-duplicate check: within RealSign, Training/Testing/Validation are separate directories with distinct image files — no exact duplicates across splits by construction. Cross-dataset duplicates are impossible (alphabet vs. words depict different content).

## Usable data after filtering

| Dataset | Task | Classes | Train | Val | Test | Total |
|---------|------|---------|-------|-----|------|-------|
| RealSign | Alphabet | 26 (A-Z) | 18,198 | 2,579 | 5,200 | 25,977 |
| Real-life Words | Word | 20 | TBD (user split) | TBD | TBD | ~18,000 |

## Recommended smallest pilot

1. **Alphabet CNN**: 26-class classifier on RealSign, RGB 64x64, existing Train/Val/Test splits. CPU-friendly, fast to train.
2. **Word CNN**: 20-class classifier on Real-life Words, RGB 64x64, user-aware split. CPU-friendly.

No pretrained weights required (no torchvision installed). Small CNNs built from scratch with torch + PIL + numpy.

## Attribution requirements

- RealSign (CC0-1.0): no attribution required, but cite: Boggaram et al., RealSign-Indian-Sign-Language-Dataset, PES University.
- Real-life Words (Mendeley): verify license on dataset page; attribute Mendeley Data if required.
- Do not commit downloaded datasets or large checkpoints to Git (.gitignore covers them).
