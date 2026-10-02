# GestureTalk ideas vs SignSage — hypothesis audit (2026-10-02)

Reference: "GestureTalk: Real-Time Sign Language Recognition Using Deep
Learning" (ResearchGate fig. 404317403). Its experiments are **isolated,
closed-set recognition** (26 static alphabet classes + 20 dynamic words) —
NOT continuous sentence translation. SignSage stays a continuous
pose→English task; nothing below converts it into an alphabet classifier.
Each idea is judged against repo evidence as ADOPT / KEEP / EXPERIMENT.

## Verdicts

1. **Landmark normalization — KEEP (already present, verified).**
   Hip-center (landmarks 23/24) + per-sequence body-std scale,
   `pose_normalize_v1`, byte-identical train/serve (direct diff of
   `normalize_pose_sequence` vs `app/ml/preprocessing/normalization.py`).
2. **Wrist-centered coordinates — EXPERIMENT (v3, not v2).**
   Wrist-centering suits isolated hand-shape classification; continuous
   signing needs torso-relative signing space, which hip-centering preserves.
   Switching representations would break checkpoint + serving parity, so it
   stays a controlled v3 ablation, never a silent v2 change.
3. **Palm/scale normalization — KEEP current scale; palm-norm is EXPERIMENT.**
   Same parity reasoning as (2).
4. **Temporal smoothing — EXPERIMENT.** Training and serving both use raw
   per-frame landmarks (no smoothing, no interpolation; missing → zeros +
   `nan_to_num`). Smoothing smears fast finger motion and changes the
   representation — v3 experiment with retraining, not a serving-only tweak.
5. **Missing-landmark interpolation — EXPERIMENT.** Same parity constraint.
   Current zero-fill is identical both sides, which is what matters for v2.
6. **Temporal windows — KEEP 160, explicitly NOT 32–64.**
   GestureTalk's short windows fit isolated words. iSign targets average ~11
   words spanning whole sentences; the pipeline's 160-frame cap with uniform
   linspace downsampling + zero-pad + padding masks is determined by that
   task, and serving reuses it (rolling 80/20 within the same 160 budget).
   Blindly copying 32–64 frames would truncate sentences.
7. **Temporal augmentation — OPEN (recommended v2 addendum).**
   Verified absent from training (notebook grep: no warp/noise/flip/drop).
   Time-masking and temporal jitter are representation-preserving and
   directly attack prior-dominance; worth adding to `train_sign_to_text.py`
   before the Kaggle run if cheap to implement correctly. Not yet implemented.
8. **Temporal attention — PRESENT.** Encoder self-attention over the frame
   sequence + decoder cross-attention is the temporal model; depthwise
   Conv1d-k5 adds local context. No change.
9. **Signer-independent evaluation — OPEN.**
   Split is leak-safe by `video_id` (asserted disjoint), but signer IDs were
   never checked: the same signer may span videos. `audit_dataset.py` now
   reports per-split signer overlap when a signer column exists; the Kaggle
   run must confirm the CSV carries one and record the result.
10. **Robustness testing — PARTIAL.** Transport hostile-input tests exist and
    pass (NaN/ragged/oversize/unknown). *Model* robustness (noise, occlusion,
    signer shift) has no harness — documented open item, not a gate change.
11. **Latency instrumentation — PRESENT.** Measured CPU: ~50 ms (v1 size) /
    ~69 ms (v2 size) per 80-frame interim; 17.2 ms live tiny-model boundary.
    2 s cadence ⇒ <5% overhead. Full browser-pipeline latency still needs a
    real-device session.
12. **Motion/velocity features — PRESENT, no dim change.**
    The model concatenates per-frame deltas internally (3258 = 2×1629 consumed
    inside `FrameProjection`; recorded in the bundle contract). Phase-2
    question answered: do NOT change input dimensionality; the representation
    already carries velocity.
13. **z handling — KEEP joint scaling.** One scalar divides x/y/z together,
    preserving aspect; per-axis z-norm would be a parity-breaking v3
    experiment. NaN/Inf: training zeroes them; serving rejects them at the
    boundary (`INVALID_MESSAGE`) — both guarantee the model never sees
    non-finite input.

## Net effect on v2

No architecture/representation change adopted from GestureTalk: items 1, 6, 8,
11, 12 confirm current choices; items 2–5 are quarantined as v3 experiments
precisely because they would break train/serve parity; items 7, 9, 10 are
recorded open items (augmentation recommended, signer check wired into the
audit tool, robustness harness future work). The v2 bet remains: same proven
plumbing, pre-norm + capacity + schedule + collapse gate.
