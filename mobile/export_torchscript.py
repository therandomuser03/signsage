#!/usr/bin/env python3
"""
Export v7 PoseStitch model (encoder + T5 decoder) to TorchScript for mobile.

Produces:
  mobile/models/
    encoder_traced.pt        — visual encoder: (B,4,94,128) → (B,128,512)
    t5_decoder_step.pt       — single decoder step with KV cache
    t5_shared_embed.pt       — token→embedding lookup
    t5_lm_head.pt            — hidden→logits projection
    t5_full_greedy.pt        — combined: encoder_in → token_ids (if scripting succeeds)
    t5_config.json           — T5 config for mobile-side loop
    model_state_dict.pt      — full weight dump as reference

Verification:
  mobile/verify_torchscript.py — cosine sim, token match, BLEU on ref set
"""

import sys, os, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from encoder import KeypointToVideoEncoder
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.mobile_optimizer import optimize_for_mobile


MODELS_DIR = Path(__file__).parent / "models"
CHECKPOINT = Path(__file__).parent.parent / "training/experiments/v7_pose_stitch/best_ps.pt"


def save_mobile(module, path, optimize=True):
    """Save a traced module for mobile: optimize + .ptl format."""
    path = path.with_suffix(".ptl")
    if optimize:
        try:
            module = optimize_for_mobile(module)
        except Exception as e:
            print(f"    ⚠️  optimize_for_mobile failed: {e}")
    module._save_for_lite_interpreter(str(path))
    print(f"    Saved: {path}")
    return path


def get_model():
    config = {
        "encoder": {"keypoint_dim": 4, "num_keypoints": 94,
                     "temporal_channels": [376, 256, 256],
                     "temporal_kernel": 5,
                     "transformer_embed_dim": 512,
                     "transformer_num_heads": 8,
                     "transformer_num_layers": 4,
                     "transformer_dropout": 0.1,
                     "output_dim": 512, "max_frames": 128},
        "t5": {"t5_model": "t5-small", "d_model": 512, "max_target_len": 128},
    }

    class V7Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            visual_cfg = config["encoder"].copy()
            visual_cfg["output_dim"] = config["t5"]["d_model"]
            self.visual_encoder = KeypointToVideoEncoder(visual_cfg)
            # Force eager attention — SDPA uses aten::scaled_dot_product_attention
            # which is not available on PyTorch Mobile
            t5_cfg = T5Config.from_pretrained(config["t5"]["t5_model"])
            t5_cfg._attn_implementation = "eager"
            self.t5 = T5ForConditionalGeneration.from_pretrained(
                config["t5"]["t5_model"], config=t5_cfg)
            self.t5.config.t5_model = config["t5"]["t5_model"]
            self.tokenizer = T5Tokenizer.from_pretrained(config["t5"]["t5_model"])

        def forward(self, keypoints, target_texts):
            encoder_output = self.visual_encoder.get_sequence_output(keypoints)
            label_ids, label_mask = self._encode_target(target_texts, keypoints.device)
            label_ids = label_ids.masked_fill(label_mask == 0, -100)
            return self.t5(inputs_embeds=encoder_output, labels=label_ids).loss

        def _encode_target(self, target_texts, device):
            labels = self.tokenizer(target_texts, max_length=128,
                                    padding="max_length", truncation=True, return_tensors="pt")
            ids = labels.input_ids.to(device)[:, :128]
            mask = labels.attention_mask.to(device)[:, :128]
            return ids, mask

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

    device = torch.device("cpu")  # Force CPU for mobile-compatible export
    model = V7Model(config).to(device)
    model.eval()

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    epoch = ckpt.get("epoch", "?")
    val_ps = ckpt.get("val_results", {}).get("ps_b4", "?")
    print(f"  Checkpoint epoch={epoch}  val PS-B4={val_ps}")

    return model, config


class DecoderStepWrapper(nn.Module):
    """Single T5 decoder step (used for tracing)."""
    def __init__(self, decoder, shared, lm_head):
        super().__init__()
        self.decoder = decoder
        self.shared = shared
        self.lm_head = lm_head

    def forward(self, encoder_hidden_states, input_ids, past_key_values_length=0):
        decoder_embeds = self.shared(input_ids)
        if past_key_values_length > 0:
            out = self.decoder(
                inputs_embeds=decoder_embeds,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=True,
            )
        else:
            out = self.decoder(
                inputs_embeds=decoder_embeds,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=True,
            )
        hidden = out.last_hidden_state[:, -1:, :]
        logits = self.lm_head(hidden)
        return logits, out.last_hidden_state, out.past_key_values


class T5GreedyScriptable(nn.Module):
    """
    Full greedy decoder: takes encoder_hidden_states, returns token IDs.
    Intended for torch.jit.script.
    """
    def __init__(self, decoder, shared, lm_head, pad_id, eos_id):
        super().__init__()
        self.decoder = decoder
        self.shared = shared
        self.lm_head = lm_head
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.max_length = 128

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        B = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device

        decoder_input_ids = torch.full((B, 1), self.pad_id, device=device, dtype=torch.long)
        all_ids = decoder_input_ids
        past_key_values = None

        for _ in range(self.max_length):
            embeds = self.shared(decoder_input_ids)
            if past_key_values is not None:
                out = self.decoder(
                    inputs_embeds=embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                out = self.decoder(
                    inputs_embeds=embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            hidden = out.last_hidden_state[:, -1:, :]
            logits = self.lm_head(hidden)
            next_token = logits.argmax(dim=-1)
            all_ids = torch.cat([all_ids, next_token], dim=1)
            decoder_input_ids = next_token
            if next_token.item() == self.eos_id:
                break

        return all_ids


def export_visual_encoder(model, device):
    """Trace visual_encoder.get_sequence_output with fixed (B,4,94,128) input."""
    print("\n── Exporting Visual Encoder ──")
    example = torch.randn(1, 4, 94, 128, device=device)
    class EncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self, keypoints):
            return self.encoder.get_sequence_output(keypoints)

    wrapper = EncoderWrapper(model.visual_encoder).to(device).eval()
    traced = torch.jit.trace(wrapper, example, check_trace=False)
    traced_path = MODELS_DIR / "encoder_traced.pt"
    traced.save(str(traced_path))
    save_mobile(traced, MODELS_DIR / "encoder_traced.pt")
    out = traced(example)
    print(f"  Input:  {tuple(example.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Saved:  {traced_path}")

    # Verify vs original
    with torch.no_grad():
        ref = model.visual_encoder.get_sequence_output(example)
    cos = nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    print(f"  Cosine sim vs original: {cos.item():.6f}")
    assert cos > 0.999, f"Visual encoder trace drift: cos={cos.item():.6f}"
    print(f"  ✅ Encoder trace verified")
    return traced


def export_decoder_components(model, device):
    """Export T5 decoder step, shared embed, and lm_head as separate traced modules."""
    print("\n── Exporting T5 Decoder Components ──")

    B, T, D = 1, 128, 512
    encoder_hidden = torch.randn(B, T, D, device=device)
    input_ids = torch.full((B, 1), model.t5.config.pad_token_id, device=device, dtype=torch.long)

    # 1. Shared embedding
    print("  Exporting t5_shared_embed...")

    class SharedEmbedWrapper(nn.Module):
        def __init__(self, embed):
            super().__init__()
            self.embed = embed
        def forward(self, ids):
            return self.embed(ids)

    shared_wrapper = SharedEmbedWrapper(model.t5.shared).to(device).eval()
    traced_embed = torch.jit.trace(
        shared_wrapper,
        input_ids,
        check_trace=False,
    )
    embed_path = MODELS_DIR / "t5_shared_embed.pt"
    traced_embed.save(str(embed_path))
    save_mobile(traced_embed, embed_path)
    emb_out = traced_embed(input_ids)
    print(f"    Input:  {tuple(input_ids.shape)} → Output: {tuple(emb_out.shape)}")
    print(f"    Saved:  {embed_path}")

    # 2. T5 Encoder (traced: visual_encoder_out → T5_encoder_out)
    print("  Exporting t5_encoder...")
    class T5EncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self, hidden_states):
            return self.encoder(inputs_embeds=hidden_states).last_hidden_state

    t5_enc_wrapper = T5EncoderWrapper(model.t5.encoder).to(device).eval()
    traced_t5_enc = torch.jit.trace(t5_enc_wrapper, encoder_hidden, check_trace=False)
    t5_enc_path = MODELS_DIR / "t5_encoder.pt"
    traced_t5_enc.save(str(t5_enc_path))
    save_mobile(traced_t5_enc, t5_enc_path)
    t5_enc_out = traced_t5_enc(encoder_hidden)
    print(f"    Input:  {tuple(encoder_hidden.shape)} → Output: {tuple(t5_enc_out.shape)}")
    print(f"    Saved:  {t5_enc_path}")
    # Verify
    with torch.no_grad():
        ref_t5_enc = model.t5.encoder(inputs_embeds=encoder_hidden).last_hidden_state
    cos_enc = nn.functional.cosine_similarity(t5_enc_out.flatten(), ref_t5_enc.flatten(), dim=0)
    print(f"    Cosine sim vs original: {cos_enc.item():.6f}")

    # 3. Decoder step (first call, no past_key_values — returns hidden + cache)
    print("  Exporting t5_decoder_first_step...")
    class DecoderFirstStep(nn.Module):
        def __init__(self, decoder, shared):
            super().__init__()
            self.decoder = decoder
            self.shared = shared

        def forward(self, encoder_hidden_states, input_ids):
            embeds = self.shared(input_ids)
            out = self.decoder(
                inputs_embeds=embeds,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=True,
            )
            pkv = out.past_key_values
            sac = pkv.self_attention_cache
            cac = pkv.cross_attention_cache
            n = len(sac.layers)
            out_sa_keys = tuple(sac.layers[i].keys for i in range(n))
            out_sa_vals = tuple(sac.layers[i].values for i in range(n))
            out_ca_keys = tuple(cac.layers[i].keys for i in range(n))
            out_ca_vals = tuple(cac.layers[i].values for i in range(n))
            return out.last_hidden_state, out_sa_keys, out_sa_vals, out_ca_keys, out_ca_vals

    first_step = DecoderFirstStep(model.t5.decoder, model.t5.shared).to(device).eval()

    try:
        traced_first = torch.jit.trace(
            first_step,
            (encoder_hidden, input_ids),
            check_trace=False,
        )
        first_path = MODELS_DIR / "t5_decoder_first_step.pt"
        traced_first.save(str(first_path))
        save_mobile(traced_first, first_path)
        print(f"    Saved: {first_path}")
        # Verify first step output
        f_out = traced_first(encoder_hidden, input_ids)
        print(f"    Output: hidden{f_out[0].shape}, 4×6 cache tensors")
        # Cosine sim on hidden
        with torch.no_grad():
            emb = model.t5.shared(input_ids)
            ref = model.t5.decoder(inputs_embeds=emb, encoder_hidden_states=encoder_hidden, use_cache=True)
        cos_f = nn.functional.cosine_similarity(f_out[0].flatten(), ref.last_hidden_state.flatten(), dim=0)
        print(f"    Hidden cosine sim: {cos_f.item():.6f}")
    except Exception as e:
        print(f"    ⚠️  First step trace failed: {e}")

    # 3. Decoder step (with past_key_values via raw tensor tuples)
    print("  Exporting t5_decoder_step...")
    from transformers.cache_utils import EncoderDecoderCache, DynamicCache

    class DecoderStepRaw(nn.Module):
        def __init__(self, decoder, shared, lm_head):
            super().__init__()
            self.decoder = decoder
            self.shared = shared
            self.lm_head = lm_head

        def forward(self, encoder_hidden_states, input_ids,
                    sa_keys, sa_vals, ca_keys, ca_vals):
            num_layers = len(sa_keys)
            sa_cache = DynamicCache()
            ca_cache = DynamicCache()
            for i in range(num_layers):
                sa_cache.update(sa_keys[i], sa_vals[i], i)
                ca_cache.update(ca_keys[i], ca_vals[i], i)
            pkv = EncoderDecoderCache(sa_cache, ca_cache)

            embeds = self.shared(input_ids)
            out = self.decoder(
                inputs_embeds=embeds,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=pkv,
                use_cache=True,
            )
            pkv_out = out.past_key_values
            sac = pkv_out.self_attention_cache
            cac = pkv_out.cross_attention_cache
            n = len(sac.layers)
            out_sa_keys = tuple(sac.layers[i].keys for i in range(n))
            out_sa_vals = tuple(sac.layers[i].values for i in range(n))
            out_ca_keys = tuple(cac.layers[i].keys for i in range(n))
            out_ca_vals = tuple(cac.layers[i].values for i in range(n))

            hidden = out.last_hidden_state[:, -1:, :]
            logits = self.lm_head(hidden)
            return logits, out_sa_keys, out_sa_vals, out_ca_keys, out_ca_vals

    step_raw = DecoderStepRaw(model.t5.decoder, model.t5.shared, model.t5.lm_head).to(device).eval()

    # Get reference cache for example inputs
    with torch.no_grad():
        embeds = model.t5.shared(input_ids)
        first_out = model.t5.decoder(
            inputs_embeds=embeds,
            encoder_hidden_states=encoder_hidden,
            use_cache=True,
        )
    pkv_ref = first_out.past_key_values
    sa_keys = tuple(pkv_ref.self_attention_cache.layers[i].keys for i in range(6))
    sa_vals = tuple(pkv_ref.self_attention_cache.layers[i].values for i in range(6))
    ca_keys = tuple(pkv_ref.cross_attention_cache.layers[i].keys for i in range(6))
    ca_vals = tuple(pkv_ref.cross_attention_cache.layers[i].values for i in range(6))
    next_ids = torch.full((B, 1), 3, device=device, dtype=torch.long)

    try:
        traced_step = torch.jit.trace(
            step_raw,
            (encoder_hidden, next_ids, sa_keys, sa_vals, ca_keys, ca_vals),
            check_trace=False,
        )
        step_path = MODELS_DIR / "t5_decoder_step.pt"
        traced_step.save(str(step_path))
        save_mobile(traced_step, step_path)
        print(f"    Input: enc_hidden={encoder_hidden.shape}, ids={next_ids.shape}, "
              f"6 layers × 4 tensors")
        out = traced_step(encoder_hidden, next_ids, sa_keys, sa_vals, ca_keys, ca_vals)
        print(f"    Output: logits{out[0].shape}, 4×6 cache tensors")
        print(f"    Saved: {step_path}")

        # Verify
        with torch.no_grad():
            emb_next = model.t5.shared(next_ids)
            ref_out = model.t5.decoder(inputs_embeds=emb_next, encoder_hidden_states=encoder_hidden, past_key_values=pkv_ref, use_cache=True)
            ref_logits = model.t5.lm_head(ref_out.last_hidden_state[:, -1:, :])
        cos = nn.functional.cosine_similarity(out[0].flatten(), ref_logits.flatten(), dim=0)
        print(f"    Cosine sim vs original: {cos.item():.6f}")
    except Exception as e:
        print(f"    ⚠️  Step trace failed: {e}")

    # 5. LM Head
    print("  Exporting t5_lm_head...")
    hidden = torch.randn(B, 1, D, device=device)

    class LMHeadWrapper(nn.Module):
        def __init__(self, lm_head):
            super().__init__()
            self.lm_head = lm_head
        def forward(self, x):
            return self.lm_head(x)

    lm_mod = LMHeadWrapper(model.t5.lm_head).to(device).eval()
    traced_lm = torch.jit.trace(lm_mod, hidden, check_trace=False)
    lm_path = MODELS_DIR / "t5_lm_head.pt"
    traced_lm.save(str(lm_path))
    save_mobile(traced_lm, lm_path)
    print(f"    Input: {tuple(hidden.shape)} → Output: ({B}, 1, {model.t5.lm_head.out_features})")
    print(f"    Saved: {lm_path}")

    return {
        "shared_embed": embed_path,
        "lm_head": lm_path,
    }


def export_full_greedy(model, device):
    """Try to script the full greedy decoder (encoder_in → token_ids)."""
    print("\n── Exporting Full Greedy Decoder ──")

    greedy_mod = T5GreedyScriptable(
        model.t5.decoder, model.t5.shared, model.t5.lm_head,
        model.t5.config.pad_token_id, model.t5.config.eos_token_id,
    ).to(device).eval()

    try:
        scripted = torch.jit.script(greedy_mod)
        greedy_path = MODELS_DIR / "t5_full_greedy.pt"
        scripted.save(str(greedy_path))
        print(f"  ✅ Full greedy script successful")
        print(f"  Saved: {greedy_path}")

        # Smoke test
        enc_hidden = torch.randn(1, 128, 512, device=device)
        out = scripted(enc_hidden)
        print(f"  Smoke test: encoder_hidden(1,128,512) → token_ids{tuple(out.shape)}")
        return scripted
    except Exception as e:
        print(f"  ⚠️  Full greedy scripting failed: {e}")
        return None


def export_config(model):
    """Save T5 config for mobile-side generation loop."""
    config = {
        "vocab_size": model.t5.config.vocab_size,
        "d_model": model.t5.config.d_model,
        "d_kv": model.t5.config.d_kv,
        "num_layers": model.t5.config.num_layers,
        "num_decoder_layers": model.t5.config.num_decoder_layers,
        "num_heads": model.t5.config.num_heads,
        "d_ff": model.t5.config.d_ff,
        "pad_token_id": model.t5.config.pad_token_id,
        "eos_token_id": model.t5.config.eos_token_id,
        "decoder_start_token_id": model.t5.config.decoder_start_token_id,
        "max_target_len": 128,
    }
    config_path = MODELS_DIR / "t5_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved: {config_path}")
    return config


def export_full_state_dict(model):
    """Save full state dict as reference."""
    sd_path = MODELS_DIR / "model_state_dict.pt"
    torch.save(model.state_dict(), sd_path)
    print(f"  Reference state dict saved: {sd_path}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # Force CPU — mobile has no CUDA
    print(f"Device: {device} (forced CPU for mobile-compatible export)")
    print(f"Checkpoint: {CHECKPOINT}")

    model, config = get_model()

    export_full_state_dict(model)
    export_config(model)

    traced_encoder = export_visual_encoder(model, device)
    export_decoder_components(model, device)
    export_full_greedy(model, device)

    print("\n── Export Summary ──")
    for f in sorted(MODELS_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:30s} {size_mb:8.2f} MB")
    print(f"\n  ✅ Export complete. Models in {MODELS_DIR}/")
    print(f"\n  NOTE: t5_full_greedy.pt failed — T5 internals not scriptable.")
    print(f"  For mobile, use the 6 components + manual loop:")
    print(f"    1. encoder_traced.pt       keypoints → vis_enc_out (B,128,512)")
    print(f"    2. t5_encoder.pt           vis_enc_out → t5_enc_out (B,128,512)")
    print(f"    3. t5_shared_embed.pt      token_id → embedding (B,1,512)")
    print(f"    4. t5_decoder_first_step  1st decoder step → (hidden, 4×6 cache)")
    print(f"    5. t5_decoder_step.pt      subsequent steps → (logits, 4×6 cache)")
    print(f"    6. t5_lm_head.pt           hidden → logits (B,1,32128)")
    print(f"    Loop: encoder → t5_encoder → first_step → lm_head → argmax → "
          f"step(cache) → lm_head → argmax ... <eos>")


if __name__ == "__main__":
    main()
