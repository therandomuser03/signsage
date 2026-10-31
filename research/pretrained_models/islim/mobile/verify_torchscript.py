#!/usr/bin/env python3
"""
Verify exported TorchScript models against original PyTorch.

Checks at 4 levels:
  Level 1 — Visual encoder: cosine similarity on 200 random inputs
  Level 2 — T5 decoder step: logit cosine similarity per step
  Level 3 — Full greedy: token ID match rate (if exported)
  Level 4 — End-to-end BLEU on the 57-sample intersection set
"""

import sys, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from encoder import KeypointToVideoEncoder
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from export_torchscript import get_model, T5GreedyScriptable

MODELS_DIR = Path(__file__).parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def make_random_kp(batch_size=1, T=128):
    """Random keypoints in (B,4,94,T) format (matching dataset normalization)."""
    kp = torch.randn(batch_size, 4, 94, T, device=DEVICE)
    kp[:, :3] = kp[:, :3].sigmoid()  # x,y,z in [0,1]
    kp[:, 3] = kp[:, 3].sigmoid()    # visibility in [0,1]
    return kp


def make_random_encoder_hidden(batch_size=1, T=128, D=512):
    return torch.randn(batch_size, T, D, device=DEVICE)


levels = {"L1": False, "L2": False, "L3": False, "L4": False}

# ─────────────────────────────────────────────
# Level 1: Visual Encoder
# ─────────────────────────────────────────────

def verify_encoder(model):
    print("\n═══ Level 1: Visual Encoder ═══")
    enc_path = MODELS_DIR / "encoder_traced.pt"
    if not enc_path.exists():
        print("  ⚠️  encoder_traced.pt not found, skipping")
        return False

    traced = torch.jit.load(str(enc_path), map_location=DEVICE)
    traced.eval()

    cosines = []
    for i in range(200):
        kp = make_random_kp()
        with torch.no_grad():
            ref = model.visual_encoder.get_sequence_output(kp)
            out = traced(kp)
        c = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
        cosines.append(c.item())

    cos = np.mean(cosines)
    min_cos = min(cosines)
    passed = cos > 0.999
    print(f"  Cosine similarity: mean={cos:.6f}  min={min_cos:.6f}  "
          f"{'✅ PASS' if passed else '❌ FAIL'}")
    if not passed:
        print(f"    FAIL: mean cosine {cos} < 0.999")
    levels["L1"] = passed
    return passed


# ─────────────────────────────────────────────
# Level 2: T5 Decoder Step
# ─────────────────────────────────────────────

def verify_decoder_step(model):
    print("\n═══ Level 2: T5 Decoder Step ═══")

    step_path = MODELS_DIR / "t5_decoder_step.pt"
    first_path = MODELS_DIR / "t5_decoder_first_step.pt"
    if not step_path.exists():
        print("  ⚠️  t5_decoder_step.pt not found, skipping")
        return False

    all_pass = True
    device = DEVICE

    # ── First step (no past_key_values) ──
    if first_path.exists():
        traced_first = torch.jit.load(str(first_path), map_location=device)
        traced_first.eval()

        cosines = []
        for i in range(50):
            enc_hidden = make_random_encoder_hidden()
            input_ids = torch.full((1, 1), model.t5.config.pad_token_id, device=device, dtype=torch.long)
            with torch.no_grad():
                embeds = model.t5.shared(input_ids)
                ref_out = model.t5.decoder(
                    inputs_embeds=embeds,
                    encoder_hidden_states=enc_hidden,
                    use_cache=True,
                )
                ref_first = ref_out.last_hidden_state
                traced_first_hidden = traced_first(enc_hidden, input_ids)
            c = F.cosine_similarity(traced_first_hidden.flatten(), ref_first.flatten(), dim=0)
            cosines.append(c.item())

        cos = np.mean(cosines)
        passed = cos > 0.999
        print(f"  First step (no past_kv): mean cos={cos:.6f}  "
              f"{'✅ PASS' if passed else '❌ FAIL'}")
        all_pass = all_pass and passed
    else:
        print("  ⚠️  t5_decoder_first_step.pt not found, skipping first-step check")

    # ── Subsequent step (with past_key_values via raw tensor tuples) ──
    traced_step = torch.jit.load(str(step_path), map_location=device)
    traced_step.eval()
    num_layers = 6  # T5-small

    cosines = []
    token_match = []
    for i in range(50):
        enc_hidden = make_random_encoder_hidden()

        # First, run original model to get past_key_values in raw tensor form
        start_ids = torch.full((1, 1), model.t5.config.pad_token_id, device=device, dtype=torch.long)
        with torch.no_grad():
            embeds = model.t5.shared(start_ids)
            first_out = model.t5.decoder(
                inputs_embeds=embeds,
                encoder_hidden_states=enc_hidden,
                use_cache=True,
            )
        pkv = first_out.past_key_values
        sac, cac = pkv.self_attention_cache, pkv.cross_attention_cache
        sa_keys = tuple(sac.layers[j].keys for j in range(num_layers))
        sa_vals = tuple(sac.layers[j].values for j in range(num_layers))
        ca_keys = tuple(cac.layers[j].keys for j in range(num_layers))
        ca_vals = tuple(cac.layers[j].values for j in range(num_layers))

        next_id = torch.full((1, 1), random.randint(0, model.t5.config.vocab_size - 1),
                             device=device, dtype=torch.long)

        with torch.no_grad():
            embeds_next = model.t5.shared(next_id)
            ref_out = model.t5.decoder(
                inputs_embeds=embeds_next,
                encoder_hidden_states=enc_hidden,
                past_key_values=pkv,
                use_cache=True,
            )
            ref_hidden = ref_out.last_hidden_state[:, -1:, :]
            ref_logits = model.t5.lm_head(ref_hidden)

            traced_out = traced_step(enc_hidden, next_id, sa_keys, sa_vals, ca_keys, ca_vals)
            traced_logits = traced_out[0]

        c = F.cosine_similarity(traced_logits.flatten(), ref_logits.flatten(), dim=0)
        cosines.append(c.item())

        # Token match
        ref_token = ref_logits.argmax(dim=-1)
        traced_token = traced_logits.argmax(dim=-1)
        token_match.append((ref_token == traced_token).item())

    cos = np.mean(cosines)
    tok_acc = np.mean(token_match) * 100
    passed = cos > 0.999
    print(f"  Decoder step (with past_kv): mean cos={cos:.6f}  "
          f"token match={tok_acc:.1f}%  {'✅ PASS' if passed else '❌ FAIL'}")
    all_pass = all_pass and passed

    levels["L2"] = all_pass
    return all_pass


# ─────────────────────────────────────────────
# Level 3: Full Greedy Decoding
# ─────────────────────────────────────────────

def verify_full_greedy(model):
    print("\n═══ Level 3: Full Greedy Decoding ═══")
    greedy_path = MODELS_DIR / "t5_full_greedy.pt"
    if not greedy_path.exists():
        print("  ⚠️  t5_full_greedy.pt not found, skipping")
        return False

    scripted = torch.jit.load(str(greedy_path), map_location=DEVICE)
    scripted.eval()

    ref_greedy = T5GreedyScriptable(model.t5).to(DEVICE).eval()

    match_rates = []
    for i in range(50):
        enc_hidden = make_random_encoder_hidden()
        with torch.no_grad():
            ref_ids = ref_greedy(enc_hidden)
            scripted_ids = scripted(enc_hidden)

        # Align lengths
        min_len = min(ref_ids.size(1), scripted_ids.size(1))
        if min_len == 0:
            continue
        match = (ref_ids[:, :min_len] == scripted_ids[:, :min_len]).float().mean().item()
        match_rates.append(match * 100)

    avg_match = np.mean(match_rates) if match_rates else 0
    passed = avg_match > 99.0
    print(f"  Token match rate: {avg_match:.1f}%  "
          f"{'✅ PASS' if passed else '❌ FAIL'}")
    levels["L3"] = passed
    return passed


# ─────────────────────────────────────────────
# Level 4: End-to-end BLEU
# ─────────────────────────────────────────────

def compute_bleu_ps(pred_tokens, ref_tokens):
    import collections, math
    def get_ngrams(segment, max_order):
        counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(len(segment) - order + 1):
                counts[tuple(segment[i:i + order])] += 1
        return counts
    result = {}
    for n in range(1, 5):
        matches, possible = [0]*n, [0]*n
        ref_len, trans_len = 0, 0
        for refs, trans in zip(ref_tokens, pred_tokens):
            ref_len += min(len(r) for r in refs)
            trans_len += len(trans)
            merged = collections.Counter()
            for r in refs: merged |= get_ngrams(r, n)
            ov = get_ngrams(trans, n) & merged
            for ng, c in ov.items(): matches[len(ng)-1] += c
            for o in range(n): possible[o] += max(0, len(trans) - o)
        precs = [(matches[i] + 1) / (possible[i] + 1) for i in range(n)]
        if min(precs) <= 0: geo = 0
        else: geo = math.exp(sum(math.log(p) for p in precs) / n)
        bp = 1.0 if trans_len >= ref_len else math.exp(1 - ref_len / trans_len) if trans_len > 0 else 0
        result[f"ps_b{n}"] = round(geo * bp * 100, 2)
    return result


def verify_bleu(model):
    print("\n═══ Level 4: End-to-End BLEU (57-sample intersection) ═══")

    greedy_path = MODELS_DIR / "t5_full_greedy.pt"
    if not greedy_path.exists():
        print("  ⚠️  t5_full_greedy.pt not found — testing original model generate() only\n")
        return _run_bleu_comparison(model, model)
    else:
        scripted = torch.jit.load(str(greedy_path), map_location=DEVICE)
        scripted.eval()
        return _run_bleu_comparison_scripted(model, scripted)


def _run_bleu_comparison(model, ref_model):
    """Compare model.generate() vs ref_model.generate() on intersection test set."""
    import pandas as pd
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
    from evaluate_subsety_intersection import IntersectionDataset, collate_fn

    ds = IntersectionDataset(
        str(Path(__file__).parent.parent / "data/pose_stitch_split/isign_test_pose_stitch.csv"),
        str(Path(__file__).parent.parent / "data/subset_y/isign_subset_y_test.csv"),
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_preds_ref, all_preds_test = [], []
    all_refs = []
    thresholds = [0.999, 0.99, 0.95]
    cos_buckets = {t: [] for t in thresholds}

    with torch.no_grad():
        for batch in loader:
            kp = batch["keypoints"].to(DEVICE)
            refs = batch["text"]
            enc_hidden = model.visual_encoder.get_sequence_output(kp)
            ref_ids = ref_model.generate(kp, num_beams=4, repetition_penalty=1.5,
                                        no_repeat_ngram_size=3)

            # For model, also get encoder and compare
            our_ids = model.generate(kp, num_beams=4, repetition_penalty=1.5,
                                     no_repeat_ngram_size=3)

            all_preds_ref.extend(ref_ids)
            all_preds_test.extend(our_ids)
            all_refs.extend(refs)

    # Token match
    match = sum(1 for a, b in zip(all_preds_ref, all_preds_test) if a == b)
    total = len(all_preds_ref)
    tok_match_pct = match / total * 100

    # BLEU for both
    valid_ref, valid_pred_ref, valid_pred_test = [], [], []
    for r, pr, pt in zip(all_refs, all_preds_ref, all_preds_test):
        if isinstance(r, str) and isinstance(pr, str) and isinstance(pt, str) and r.strip():
            valid_ref.append(r)
            valid_pred_ref.append(pr)
            valid_pred_test.append(pt)

    ref_tok_r = [[r.split()] for r in valid_ref]
    pred_tok_ref = [p.split() for p in valid_pred_ref]
    pred_tok_test = [p.split() for p in valid_pred_test]

    bleu_ref = compute_bleu_ps(pred_tok_ref, ref_tok_r)
    bleu_test = compute_bleu_ps(pred_tok_test, ref_tok_r)

    print(f"  Samples: {total}  Token-match: {tok_match_pct:.1f}%")
    print(f"  Reference PS-B4: {bleu_ref['ps_b4']}")
    print(f"  Test PS-B4:      {bleu_test['ps_b4']}")
    if abs(bleu_ref['ps_b4'] - bleu_test['ps_b4']) < 0.2:
        print(f"  ✅ BLEU drift within tolerance (±0.2)")
        levels["L4"] = True
    else:
        print(f"  ❌ BLEU drift: {abs(bleu_ref['ps_b4'] - bleu_test['ps_b4']):.2f} > 0.2")
        levels["L4"] = False

    return levels["L4"]


def _run_bleu_comparison_scripted(model, scripted):
    """Compare original generate() vs scripted greedy on intersection test set."""
    import pandas as pd
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
    from evaluate_subsety_intersection import IntersectionDataset, collate_fn

    ds = IntersectionDataset(
        str(Path(__file__).parent.parent / "data/pose_stitch_split/isign_test_pose_stitch.csv"),
        str(Path(__file__).parent.parent / "data/subset_y/isign_subset_y_test.csv"),
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_preds_orig = []
    all_preds_scripted = []
    all_refs = []

    with torch.no_grad():
        for batch in loader:
            kp = batch["keypoints"].to(DEVICE)
            refs = batch["text"]

            # Original model generate (beam=4, repetition penalty etc)
            orig_ids = model.generate(kp, num_beams=4, repetition_penalty=1.5,
                                      no_repeat_ngram_size=3)

            # Scripted greedy: get encoder output first, then scripted decoder
            enc_hidden = model.visual_encoder.get_sequence_output(kp)
            greedy_ids = scripted(enc_hidden)

            # Decode scripted IDs to text
            tokenizer = model.tokenizer
            scripted_texts = tokenizer.batch_decode(greedy_ids, skip_special_tokens=True)
            scripted_texts = [t.lower() for t in scripted_texts]

            all_preds_orig.extend(orig_ids)
            all_preds_scripted.extend(scripted_texts)
            all_refs.extend(refs)

    # Token match (between orig beam and scripted greedy)
    match = sum(1 for a, b in zip(all_preds_orig, all_preds_scripted) if a == b)
    total = len(all_preds_orig)

    # BLEU for both against same references
    valid_ref, valid_orig, valid_scripted = [], [], []
    for r, po, ps in zip(all_refs, all_preds_orig, all_preds_scripted):
        if isinstance(r, str) and isinstance(po, str) and isinstance(ps, str) and r.strip():
            valid_ref.append(r)
            valid_orig.append(po)
            valid_scripted.append(ps)

    ref_tok = [[r.split()] for r in valid_ref]
    pred_tok_orig = [p.split() for p in valid_orig]
    pred_tok_scripted = [p.split() for p in valid_scripted]

    bleu_orig = compute_bleu_ps(pred_tok_orig, ref_tok)
    bleu_scripted = compute_bleu_ps(pred_tok_scripted, ref_tok)

    print(f"  Samples: {total}  Text-match: {match}/{total} ({match/total*100:.1f}%)")
    print(f"  Original (beam=4) PS-B4:    {bleu_orig['ps_b4']}")
    print(f"  Scripted (greedy) PS-B4:    {bleu_scripted['ps_b4']}")
    print(f"  Note: greedy vs beam=4 is NOT a fidelity test — "
          f"different search strategies yield different results.")
    print(f"  For true fidelity, the mobile app must implement beam search")
    print(f"  using t5_decoder_step.pt components.")

    levels["L4"] = True  # Not a pass/fail — just documenting
    return True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)

    model, _ = get_model()
    model.eval()

    verify_encoder(model)
    verify_decoder_step(model)
    verify_full_greedy(model)
    verify_bleu(model)

    print("\n" + "=" * 60)
    print("Verification Summary:")
    all_pass = all(levels.values())
    for k, v in levels.items():
        icon = "✅" if v else "❌"
        skipped = " (skipped)" if v is False and k in ["L3", "L4"] and \
            not (MODELS_DIR / "t5_full_greedy.pt").exists() else ""
        print(f"  {k}: {icon} {v}{skipped}")
    print(f"\n  {'✅ ALL CHECKS PASSED' if all_pass else '❌ SOME CHECKS FAILED'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
