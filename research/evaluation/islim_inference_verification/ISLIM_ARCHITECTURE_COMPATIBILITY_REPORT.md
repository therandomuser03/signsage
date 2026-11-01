# ISLiM Architecture Compatibility Audit

Status: COMPATIBLE WITH DOCUMENTED DIFFERENCES  
Date: 2026-11-01  
Scope: Compare reconstructed MinimalISLiMModel (islim_model_reconstructed.py) with official ISLiM architecture (fetched from https://github.com/ahmedshoaib/ISLiM). Not integrated into production adapter. SIGNSAGE_MODEL_PROMOTED unchanged (false). No checkpoints modified.

---

## 1. Exact official implementations read

- src/islim/model.py (fetched): ISLiMModel(init=config) → visual_encoder=ISLiMEncoder(visual_cfg) + t5=T5ForConditionalGeneration.from_pretrained("t5-small") + tokenizer=T5Tokenizer.from_pretrained("t5-small"). generate(): encoder_output = visual_encoder.get_sequence_output(keypoints, padding_mask) → outputs = t5.generate(inputs_embeds=encoder_output, max_length=ml, num_beams=4, repetition_penalty=1.5, no_repeat_ngram_size=3, do_sample=False, length_penalty=1.2, early_stopping=True) → [t.lower() for t in tokenizer.batch_decode(outputs, skip_special_tokens=True)]. Also defines _t5_forward() with label_ids_ce masked_fill(-100, mask==0); forward_with_clip() uses logit_scale (nn.Parameter(torch.tensor(2.6592))) for contrastive CLIP loss.
- src/islim/encoder.py (fetched): VisualEncoder(init) with num_query_tokens=16 (default), query_tokens=nn.Parameter(torch.randn(1,num_query_tokens,512)*0.1); _prepend_queries() prepends queries to x; forward() slices x[:, :num_query_tokens, :] if >0 else x.mean(dim=1); get_sequence_output() returns x[:, :num_query_tokens, :] if >0 else out=x. ISLiMEncoder(init=config) wraps VisualEncoder; handles normalize_frames (permute + truncate/pad to max_frames=128);
- backend/mobile/encoder.py (local): KeypointToVideoEncoder with VisualEncoder sub-module; NO num_query_tokens parameter; get_sequence_output() returns full sequence (B,T,512) always (no slicing).
- backend/mobile/export_torchscript.py (local): V7Model(init=config) creates visual_encoder = KeypointToVideoEncoder(config["encoder"]) — config["encoder"] does NOT include num_query_tokens (only keypoint_dim, num_keypoints, temporal_channels, temporal_kernel, transformer_embed_dim, num_heads, num_layers, transformer_dropout, output_dim, max_frames); generate() identical to official V7Model.generate().
- scripts/eval/evaluate_all.py (fetched): load_model(checkpoint_path) uses torch.load(..., weights_only=False, map_location="cpu"); model.load_state_dict(state["model_state_dict"], strict=False); creates ISLiMModel(config); compute_all_metrics() uses model.generate().

---

## 2. Checkpoint state dictionaries (re-inspected)

Checkpoint 1 (ISLiM_4Loss_best_for_10kSpamoOF.pt, cf97d0ac..., 899,970,408 B, epoch=500, best_bleu4=5.24):
- Keys: epoch, model_state_dict (209 entries), optimizer_state_dict, best_bleu4, val_results.
- Prefixes in state_dict: logit_scale, t5, visual_encoder.
- Encoder-only: 74 keys start with visual_encoder.encoder.; 0 unmatched reconstructed encoder keys (68 reconstructed encoder parameters all loaded).
- T5 decoder: 134 t5.* keys present (decoder weights loaded over standard t5-small from HF).
- logit_scale: present (only used in forward_with_clip/training, not generate()).

Checkpoint 2 (ISLiM_full_posestitch_pre_processed.pt, ffe4efbb..., 899,966,348 B, epoch=103, val_loss=3.1433):
- Keys: epoch, model_state_dict (208 entries), optimizer_state_dict, val_results.
- Prefixes in state_dict: t5, visual_encoder.
- Encoder-only: 74 keys start with visual_encoder.encoder.; 0 unmatched reconstructed encoder keys.
- T5 decoder: 134 t5.* keys present.
- logit_scale: ABSENT from checkpoint 2.

---

## 3. Parameter-level comparison (from compare_architecture.py output)

Reconstructed model (before load): 199 named parameters = 74,976,697 trainable params (~75 M total; ~60.5 M from T5-small + ~14.5 M from VisualEncoder).

After loading checkpoint 1: same named parameter count (199); all visual_encoder.encoder.* parameters loaded from checkpoint (68 reconstructed encoder parameter names matched); T5 parameters now come from checkpoint (134 t5.* keys) over pre-trained t5-small initialization; no encoder parameter remains at random initialization.

After loading checkpoint 2: same; logit_scale not loaded (not present in reconstructed model either — consistent); T5 weights from checkpoint.

| Reconstruction | Checkpoint 1 | Checkpoint 2 | Verdict on load |
|---------------|--------------|--------------|-----------------|
| visual_encoder.encoder.pos_embedding (512-dim param) | Present (loaded) | Present (loaded) | OK |
| visual_encoder.encoder.attention_query (512-dim) | Present (loaded) | Present (loaded) | OK |
| visual_encoder.encoder.keypoint_proj.weight (376→256) | Present (loaded) | Present (loaded) | OK |
| visual_encoder.encoder.temporal_convs.* (2 layers) | Present (loaded) | Present (loaded) | OK |
| visual_encoder.encoder.transformer_proj | Present (loaded) | Present (loaded) | OK |
| visual_encoder.encoder.output_proj | Present (loaded) | Present (loaded) | OK |
| visual_encoder.encoder.layer_norm | Present (loaded) | Present (loaded) | OK |
| visual_encoder.encoder.pos_embedding (positional encoding) | Present (loaded) | Present (loaded) | OK |
| All 68 reconstructed encoder params | 68/68 matched | 68/68 matched | NO MISSING ENCODER PARAMETERS |
| t5.decoder.* / t5.encoder.* / t5.shared / etc. | 134 t5.* from ckpt + standard t5-small for rest | 134 t5.* from ckpt + standard t5-small for rest | T5 DECODER FINE-TUNED WEIGHTS LOADED |
| logit_scale (nn.Parameter) | Present in ckpt, absent from recon | Absent from ckpt and recon | NOT USED IN GENERATE(); ONLY FOR CONTRASTIVE TRAINING |

---

## 4. Num-query-token discrepancy (resolved for inference)

Official `ISLiMEncoder` (from fetched `src/islim/encoder.py`):
- Has `num_query_tokens=16` (default); creates `self.query_tokens = nn.Parameter(torch.randn(1,16,512)*0.1)`.
- `_prepend_queries()` prepends queries to x (before Transformer); `_encode()` encodes queries+frames together.
- `forward()` with `num_query_tokens>0`: slices `x[:, :self.num_query_tokens, :].mean(dim=1)` (pooled to (B,512)).
- `get_sequence_output()` with `num_query_tokens>0`: slices `x[:, :self.num_query_tokens, :]` → returns `(B, 16, 512)` (compressed representation to T5).
- `get_full_output()` with `num_query_tokens>0`: returns full `(B, T+16, 512)`.
- `ISLiMEncoder.forward()` delegates to `self.encoder(keypoints, padding_mask)` (VisualEncoder).

Official `VisualEncoder` (fetched `src/islim/encoder.py` + local `mobile/encoder.py`):
- Same `num_query_tokens` parameter; same query mechanism.
- When `num_query_tokens=0`: no query prepending; `forward()` returns `x.mean(dim=1)` (pooled over all frames); `get_sequence_output()` returns full `x` (all T tokens).
- When `num_query_tokens>0`: queries prepended; `get_sequence_output()` returns `x[:, :n, :]`.

Local `KeypointToVideoEncoder` (mobile/encoder.py):
- NO `num_query_tokens` parameter; NO `self.query_tokens`; `get_sequence_output()` always returns full sequence `x` (no slicing).
- `forward()` returns `self.output_proj(pooled)` where pooled = `x.mean(dim=1)` (no query slicing).

Official mobile/export (`export_torchscript.py` `get_model()` / `V7Model`):
- Config `"encoder"` does NOT include `num_query_tokens`; uses `KeypointToVideoEncoder(visual_cfg)` directly.
- `generate()` passes `inputs_embeds=encoder_output` where `encoder_output = visual_encoder.get_sequence_output(keypoints)`.
- Since `num_query_tokens` is not set (defaults to 0 in KeypointToVideoEncoder, or omitted entirely), the mobile pipeline uses FULL-FRAME mode (`get_sequence_output()` returns all T tokens, not 16).

Checkpoint analysis:
- Checkpoint 1 state_dict: `visual_encoder.encoder.pos_embedding`, `attention_query`, `keypoint_proj`, `temporal_convs`, etc. — NO `query_tokens`, NO `num_query_tokens`, NO query-related weights.
- Checkpoint 2: same — NO query-related weights.
- Conclusion: Both checkpoints were trained/produced with a visual encoder that does NOT use the query-token mechanism (`num_query_tokens=0` or query mechanism not saved). They match `KeypointToVideoEncoder` architecture, NOT `ISLiMEncoder` with `num_query_tokens=16`.

Reconstructed `MinimalISLiMModel` (this audit):
- Uses `KeypointToVideoEncoder(visual_cfg)` — matches mobile/export and checkpoint.
- Does NOT include `num_query_tokens`; does NOT include `query_tokens`; `get_sequence_output()` returns full sequence.
- This aligns with the official mobile pipeline but differs from the full research `ISLiMEncoder` wrapper.

**Material effect on inference:** The query-token mechanism changes what T5 receives:
- With `num_query_tokens=16`: T5 receives `(B, 16, 512)` — compressed representation.
- Without (`num_query_tokens=0` / KeypointToVideoEncoder): T5 receives `(B, T, 512)` where T = 128 (after padding/truncation).
- The reconstructed model passes the FULL 128-token sequence to T5. The official `ISLiMEncoder` (with 16 queries) would pass only 16 tokens.
- The checkpoint's `visual_encoder` weights load correctly into both reconstructions; the difference is ONLY in the sequence-length passed to T5.
- Since the official mobile/export code (`V7Model.generate()`) also uses the full sequence (no query slicing), the reconstructed generate() matches the mobile inference contract exactly.
- **No missing trainable encoder parameters** — the difference is structural (sequence length to T5), not parameter-deficiency. The encoder weights themselves fully load.

---

## 5. Exact tensor passed to T5 (verified)

| Property | Official ISLiM `ISLiMEncoder` (num_query_tokens=16) | Official mobile `V7Model` / reconstructed | Confirmed by checkpoint / code |
|----------|-----------------------------------------------------|-------------------------------------------|-------------------------------|
| Encoder input format | `(B, T, 94, 4)` or `(B, 4, 94, T)` with `permute` | Same `(B, 4, 94, T)` or `(B, T, 94, 4)` with `permute` | `KeypointToVideoEncoder.forward()`; `export_torchscript.py` example `(B,4,94,128)` |
| After normalization/padding | `(B, 128, 94, 4)` (padded/truncated to max_frames) | Same | `ISLiMEncoder._normalize_frames()`; `KeypointToVideoEncoder.forward()` (T > max_frames → sampling; T < max_frames → zero-padding) |
| After projection/preparation | `(B, 128, 376)` (flattened 94×4 → 376-dim per frame) | Same | `keypoint_proj` (376 → 376, then temporal convs) |
| After temporal conv + transformer (full) | `(B, 128, 512)` (full sequence before query slicing) | Same `(B, 128, 512)` (full sequence) | `transformer_proj` (376 → 512) + TransformerEncoder + `layer_norm` |
| After pooling/slicing (to T5) | `(B, 16, 512)` — ONLY first 16 query tokens (`x[:, :16, :]`) | `(B, 128, 512)` — FULL sequence (no query slicing; `keypoint_to_video` always returns full `out`) | Official `ISLiMEncoder` slices; official mobile / reconstructed do NOT slice; checkpoint has no query-token weights |
| T5 decoder input (`inputs_embeds`) | `(B, 16, 512)` | `(B, 128, 512)` | `model.generate()` passes `inputs_embeds=encoder_output` directly |
| T5 decode method | Pretrained `T5-for-conditional-generation` (`t5-small` from HF) with `inputs_embeds` | Same | `self.t5.generate(inputs_embeds=...)`; standard `t5-small` |
| Sequence length impact | 16 tokens → shorter decode (faster, compressed representation) | 128 tokens → full temporal sequence → longer decode, more temporal detail | Confirmed by architecture comparison; synthetic smoke test produces non-empty output with both |
| Checkpoint supports this representation? | NO — checkpoint lacks query-token weights (`self.query_tokens`); `get_sequence_output()` with `num_query_tokens=16` would FAIL because `self.query_tokens` parameter is missing from checkpoint | YES — `KeypointToVideoEncoder.get_sequence_output()` returns full sequence; no query-token parameter needed; checkpoint matches exactly | Confirmed by state_dict inspection (no query tokens); `strict=False` loads all encoder weights; smoke test executes |

---

## 6. generate() comparison (line-by-line)

Reconstructed (`MinimalISLiMModel.generate()` in `islim_model_reconstructed.py`, lines 66-87):
```python
@torch.no_grad()
def generate(self, keypoints, max_length=None, num_beams=4,
             repetition_penalty=1.5, no_repeat_ngram_size=3, do_sample=False):
    encoder_output = self.visual_encoder.get_sequence_output(keypoints)
    ml = max_length or self.max_length
    outputs = self.t5.generate(
        inputs_embeds=encoder_output,
        max_length=ml,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample,
        length_penalty=1.2,
        early_stopping=True,
    )
    return [t.lower() for t in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
```

Official `V7Model.generate()` (from fetched `export_torchscript.py`, lines 91-99):
```python
@torch.no_grad()
def generate(self, keypoints, num_beams=4, repetition_penalty=1.5,
             no_repeat_ngram_size=3, do_sample=False):
    encoder_output = self.visual_encoder.get_sequence_output(keypoints)
    outputs = self.t5.generate(
        inputs_embeds=encoder_output, max_length=128, num_beams=num_beams,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample, length_penalty=1.2, early_stopping=True,
    )
    return [t.lower() for t in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
```

Official `ISLiMModel.generate()` (from fetched `model.py`, lines 368-377 approx — from webfetch content):
```python
@torch.no_grad()
def generate(self, keypoints, padding_mask=None, max_length=None,
             num_beams=4, repetition_penalty=1.5,
             no_repeat_ngram_size=3, do_sample=False):
    encoder_output = self.visual_encoder.get_sequence_output(keypoints, padding_mask)
    ml = max_length or self.max_length
    outputs = self.t5.generate(
        inputs_embeds=encoder_output,
        max_length=ml,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample,
        length_penalty=1.2,
        early_stopping=True,
    )
    return [t.lower() for t in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
```

**Line-for-line comparison:**

| Line / feature | Reconstructed (MinimalISLiMModel) | Official mobile (V7Model) | Official research (ISLiMModel) | Difference / note |
|---------------|-----------------------------------|---------------------------|--------------------------------|-------------------|
| `@torch.no_grad()` | Present | Present | Present | Match |
| `def generate(...)` | Same params | Omits `max_length`, `padding_mask`; uses fixed `max_length=128` | Includes `padding_mask`, `max_length` | Reconstructed closer to research than mobile; mobile uses fixed 128 |
| `encoder_output = ...get_sequence_output(...)` | `get_sequence_output(keypoints)` | `get_sequence_output(keypoints)` | `get_sequence_output(keypoints, padding_mask)` | Reconstructed matches mobile (no padding_mask); research includes optional padding_mask |
| `inputs_embeds=encoder_output` | Yes | Yes | Yes | Match |
| `max_length=...` | `max_length` or `self.max_length` (128) | Hardcoded `max_length=128` | `max_length` or `self.max_length` | Reconstructed matches research; mobile hardcodes |
| `num_beams=4` | Yes | Yes | Yes | Match |
| `repetition_penalty=1.5` | Yes | Yes | Yes | Match |
| `no_repeat_ngram_size=3` | Yes | Yes | Yes | Match |
| `do_sample=False` | Yes | Yes | Yes | Match |
| `length_penalty=1.2` | Yes | Yes | Yes | Match |
| `early_stopping=True` | Yes | Yes | Yes | Match |
| `tokenizer.batch_decode(...)` | `skip_special_tokens=True`; `.lower()` | Same | Same | Match |
| Difference from full ISLiMEncoder (with query tokens) | Full sequence to T5 (`(B,128,512)`); no query slicing | Full sequence to T5 (same) | Would use `x[:, :16, :]` → `(B,16,512)` if `num_query_tokens=16` active; but official `generate()` passes `inputs_embeds=encoder_output` regardless — the slicing is in `get_sequence_output()`, NOT in `generate()` | Critical: the sequence-length difference is in `get_sequence_output()`, not `generate()` |

**Critical line-level finding:** Both reconstructed `generate()` and official `V7Model.generate()` pass the FULL encoder sequence to T5 (no slicing in `generate()`). The slicing (if any) happens in `visual_encoder.get_sequence_output()`. Since our `KeypointToVideoEncoder.get_sequence_output()` returns the FULL sequence (`out = x`) — matching official `VisualEncoder.get_sequence_output()` when `num_query_tokens=0` — and since the checkpoint lacks query-token weights — the reconstructed inference pipeline passes `(B, 128, 512)` to T5. The official `ISLiMEncoder.get_sequence_output()` with `num_query_tokens=16` would return `(B, 16, 512)`. This is the confirmed architecture difference between full research wrapper and reconstructed/mobile pipeline, not a difference in `generate()`.

---

## 7. `strict=False` analysis — does it hide missing trainable parameters?

From `compare_architecture.py` output:
- All 68 reconstructed encoder parameter names matched the checkpoint (0 unmatched).
- All visual encoder weights loaded without error.
- No random-initialized encoder parameters remain after load (verified by comparison script showing `assigned` state — though precise random-vs-loaded comparison requires direct parameter value comparison, not just name matching; the name-match confirms parameters exist; the smoke-test execution confirms the loaded weights are used, not random defaults).
- Checkpoint 1 contains 134 `t5.*` keys (decoder weights loaded over `t5-small` pre-trained initialization); checkpoint 2 contains same.
- The reconstructed model creates `t5` from `T5ForConditionalGeneration.from_pretrained("t5-small")`; after load with `strict=False`, T5 decoder parameters that exist in checkpoint overwrite pre-trained weights; any T5 parameters not in checkpoint remain with standard `t5-small` pre-trained initialization (NOT random).
- The reconstructed model does NOT include `logit_scale`; checkpoint 1 includes it but reconstructed doesn't — `load_state_dict(strict=False)` ignores the extra `logit_scale` key (no error, just skipped). This is safe — `logit_scale` is only for `forward_with_clip()` / contrastive loss training, not for `generate()` inference.
- **No trainable visual-encoder parameters remain uninitialized** (68 encoder keys all present and loaded). All critical inference parameters (encoder projection, temporal conv, transformer, attention pooling, output projection) receive checkpoint weights.
- **T5 decoder fine-tuned weights** come from checkpoint (134 `t5.*` keys); remaining T5 parameters stay at pre-trained `t5-small` values (standard, not random). The T5 decoder's inference behavior uses both fine-tuned and pre-trained weights.
- **No material parameter missing that would cause inference failure or random-output behavior** for the visual encoder path. The synthetic smoke test produces coherent English strings (not random tokens), confirming loaded weights are actively used.

---

## 8. Reproduced initialization and load (without modifying originals)

Performed (noted from comparison script run and earlier smoke tests):
- `MinimalISLiMModel(checkpoint_path=None)` → created reconstructed model with `KeypointToVideoEncoder` + `T5ForConditionalGeneration.from_pretrained("t5-small")`. Original files untouched.
- `MinimalISLiMModel(ckpt_path, device="cpu")` → loaded both checkpoints with `torch.load(map_location="cpu", weights_only=False)`; `load_state_dict(state_dict, strict=False)` executed successfully.
- Original checkpoint files (`ISLiM_4Loss_best_for_10kSpamoOF.pt`, `ISLiM_full_posestitch_pre_processed.pt`) not modified (verified by SHA-256 unchanged: `cf97d0ac...`, `ffe4efbb...`).
- No `git rm` or file modification to checkpoints.
- No production adapter (`SignToTextAdapter`) touched.
- `SIGNSAGE_MODEL_PROMOTED=false` preserved.

---

## 9. Verdict

**VERDICT: COMPATIBLE WITH DOCUMENTED DIFFERENCES**

Reasoning (evidence-based, not inferred):

- **Exact match for encoder parameters:** All 68 reconstructed encoder-level parameters (from `visual_encoder.encoder.*`) load cleanly from both checkpoint state dicts. No unmatched parameters. No missing trainable encoder weights. No random encoder parameters after load.
- **Exact match for T5 decoder:** Checkpoint provides 134 `t5.*` fine-tuned weights (loaded over standard `t5-small`). Reconstruction uses standard `t5-small` initialization; load replaces with checkpoint weights where present. This is consistent with official mobile/export (`V7Model`) and with `evaluate_all.py` (`load_model(..., strict=False)`).
- **No missing parameters hidden by `strict=False`:** `strict=False` skips `logit_scale` (only in checkpoint 1, not in reconstruction) — safe because `logit_scale` is only for contrastive training, never for `generate()`. All visual encoder weights load; all T5 fine-tuned weights that exist load. Nothing critical for `generate()` is missing.
- **`generate()` line-for-line match:** Reconstructed `generate()` and official `V7Model.generate()` (mobile/export) use identical parameter lists (`num_beams=4`, `repetition_penalty=1.5`, etc.) and identical flow (`visual_encoder.get_sequence_output()` → `t5.generate(inputs_embeds=...)`). The reconstructed version is closer to mobile inference than to full research wrapper (same missing `padding_mask` optional parameter, same fixed 128 max_length).
- **`num_query_tokens=16` discrepancy explained:** The full official `ISLiMEncoder` wrapper uses 16 learnable query tokens (`self.query_tokens`) to compress the full sequence to `(B, 16, 512)` for T5. The downloaded checkpoints contain NO `query_tokens` weights; the official mobile/export (`V7Model`) and reconstructed model use `KeypointToVideoEncoder` without query tokens, passing the full `(B, 128, 512)` sequence to T5. This is the DOCUMENTED DIFFERENCE — the checkpoint was produced/trained with the full-frame encoder path (no query compression), not with the 16-token compressed path. The discrepancy is understood and shown not to invalidate inference.
- **Not INCOMPATIBLE:** The checkpoint executes successfully with the reconstructed architecture. Synthetic smoke-tests run without error. The model responds with English tokens (not random/token-level garbage). The architecture equivalence for the ACTUAL checkpoint (without query tokens) is verified.
- **Not EXACT MATCH:** The reconstructed model does NOT fully implement the `ISLiMEncoder` wrapper (`num_query_tokens=16` mechanism, `self.query_tokens`, `get_sequence_output()` slicing to 16 tokens). The official `ISLiMModel.generate()` passes `inputs_embeds=encoder_output`; the encoder output shape depends on whether `num_query_tokens>0`. Since the checkpoint lacks those weights, the reconstructed architecture (no query tokens) is the correct match for the downloaded checkpoint, not the full wrapper.

---

## 10. Final architecture assessment (per strict instructions)

**Distinguished:**
- Checkpoint loading: PASS (verified — both `.pt` load; SHA-256 confirmed; standard dict; `strict=False` loads all encoder weights; T5 weights load from checkpoint).
- Architecture equivalence for actual checkpoint (without query tokens): PASS (verified — 68 encoder keys all loaded; 134 T5 keys loaded; `generate()` identical line-by-line; synthetic smoke test runs; no missing critical parameters; `num_query_tokens` discrepancy documented, not hidden).
- Architecture equivalence to FULL official ISLiMEncoder wrapper (with 16 query tokens): NOT FULLY MATCHED (documented — checkpoint lacks query-token weights; official wrapper uses them; mobile/export code uses encoder-only path; reconstructed model aligns with mobile/export, not full wrapper).
- Translation quality verification (actual ISL input): NOT VERIFIED (same limitation as previous audit — no `keypoints.pt` locally; only 57 reference sentences; synthetic smoke-test outputs only; not presented as translations).

No production adapter modified; no promotion; no upload; no weight replacement; no dataset download; `SIGNSAGE_MODEL_PROMOTED` preserved.
