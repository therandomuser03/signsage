#!/usr/bin/env python3
"""Architecture compatibility audit: compare reconstructed model with official implementation."""
import sys, time, resource
from pathlib import Path
sys.path.insert(0, "/mnt/D/Codes/signsage/research/pretrained_models/islim/mobile")

import torch
import torch.nn as nn

from encoder import KeypointToVideoEncoder
from transformers import T5Config, T5ForConditionalGeneration
from islim_model_reconstructed import MinimalISLiMModel


def inspect_model_parameters(model, label):
    """Record parameter names, shapes, dtypes, and initial random state."""
    params = {}
    for name, param in model.named_parameters():
        params[name] = {
            "shape": tuple(param.shape),
            "dtype": str(param.dtype),
            "numel": param.numel(),
        }
    print(f"\n=== {label} ===")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Named parameter count: {len(params)}")
    for k in sorted(params.keys())[:20]:
        v = params[k]
        print(f"  {k}: shape={v['shape']} dtype={v['dtype']} elements={v['numel']:,}")
    return params


def main():
    ckpt1_path = "/mnt/D/Codes/signsage/research/pretrained_models/islim/checkpoints/ISLiM_4Loss_best_for_10kSpamoOF.pt"
    ckpt2_path = "/mnt/D/Codes/signsage/research/pretrained_models/islim/checkpoints/ISLiM_full_posestitch_pre_processed.pt"

    print("Phase: Reconstruct model without checkpoint")
    model_empty = MinimalISLiMModel(checkpoint_path=None, device="cpu")
    params_empty = inspect_model_parameters(model_empty, "Reconstructed model (before load)")

    for ckpt_path in [ckpt1_path, ckpt2_path]:
        print(f"\n{'='*70}")
        print(f"Loading checkpoint: {Path(ckpt_path).name}")
        start = time.time()
        model_loaded = MinimalISLiMModel(checkpoint_path=ckpt_path, device="cpu")
        load_time = time.time() - start

        # After load, inspect which parameters have changed vs empty initialization
        params_loaded = inspect_model_parameters(model_loaded, f"After loading {Path(ckpt_path).name}")

        # Check for parameters in reconstructed model that are NOT loaded from checkpoint
        # (i.e., remain at their initialized random values from KeypointToVideoEncoder / T5)
        missing_from_checkpoint = []
        loaded_ok = True
        for name in sorted(params_empty.keys()):
            v_empty = params_empty[name]
            if name not in params_loaded:
                missing_from_checkpoint.append(name)
            else:
                # Check if parameter exists in checkpoint by attempting to access state_dict
                pass
        # Instead, we know the checkpoint structure: only visual_encoder.encoder.* keys present.
        # Let's check which reconstructed sub-modules received weights.
        checkpoint_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
        checkpoint_keys = set(checkpoint_state.keys())

        # Reconstructed parameter keys (full names from MinimalISLiMModel)
        reconstructed_keys = set(params_loaded.keys())

        # Map reconstructed names to checkpoint key prefixes
        reconstructed_prefixes = set()
        for k in reconstructed_keys:
            reconstructed_prefixes.add(k.split(".")[0] if "." in k else k)

        checkpoint_prefixes = set()
        for k in checkpoint_keys:
            checkpoint_prefixes.add(k.split(".")[0] if "." in k else k)

        print(f"Checkpoint prefixes: {sorted(checkpoint_prefixes)}")
        print(f"Reconstructed prefixes: {sorted(reconstructed_prefixes)}")

        # Check overlap at encoder level
        encoder_checkpoint_keys = [k for k in checkpoint_keys if k.startswith("visual_encoder.encoder.")]
        print(f"Encoder-level checkpoint keys: {len(encoder_checkpoint_keys)}")
        print(f"Sample encoder keys: {sorted(encoder_checkpoint_keys)[:10]}")

        # Check for any unexpected/unmatched reconstructed parameters
        reconstructed_encoder_keys = [k for k in reconstructed_keys if "visual_encoder" in k]
        print(f"Reconstructed encoder-level parameter keys: {len(reconstructed_encoder_keys)}")

        # Show which reconstructed encoder keys have matching checkpoint keys
        matched = []
        unmatched = []
        for rk in reconstructed_encoder_keys:
            # Find checkpoint key that matches the reconstructed parameter
            # The mapping: reconstructed key "visual_encoder.encoder.pos_embedding"
            # matches checkpoint key "visual_encoder.encoder.pos_embedding"
            match_found = rk in checkpoint_keys  # Direct match (same name)
            if match_found:
                matched.append(rk)
            else:
                # Try partial match (e.g., reconstructed key has sub-module names that match)
                partial = False
                for ck in checkpoint_keys:
                    if rk in ck or ck in rk:
                        partial = True
                        break
                if partial:
                    matched.append(rk + " (partial/embedded match)")
                else:
                    unmatched.append(rk)

        print(f"Encoder parameter match: matched={len([m for m in matched if 'partial' not in str(m)])}, unmatched={len(unmatched)}")
        if unmatched:
            print(f"Unmatched reconstructed encoder parameters: {unmatched[:10]}")

        # Check T5 sub-module: reconstructed creates T5 from pretrained `t5-small`.
        # Checkpoint does NOT include T5 decoder weights.
        t5_checkpoint_keys = [k for k in checkpoint_keys if k.startswith("t5.") or k.startswith("self.t5.") or k.startswith("decoder.")]
        print(f"T5 decoder weights in checkpoint: {len(t5_checkpoint_keys)} (expected 0 — checkpoint is encoder-only)")

        # Resource summary
        print(f"Load time: {load_time:.2f}s")
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB (Linux reports KB)
        print(f"Peak memory (approx): ~{max_rss:.0f} MB (process-level, includes previous loads)")

        # Synthetic smoke-test only
        B, T = 1, 128
        synth = torch.randn(B, 4, 94, T, dtype=torch.float32)
        synth[:, :3, :, :] = torch.sigmoid(synth[:, :3, :, :])
        synth[:, 3, :, :] = torch.sigmoid(synth[:, 3, :, :])
        synth = synth.to("cpu")
        try:
            out = model_loaded.generate(synth)
            print(f"Synthetic smoke-test output (NOT real ISL): {out}")
            print("Smoke test: EXECUTION VERIFIED (model responds without crash).")
        except Exception as exc:
            print(f"Smoke-test generate failed: {exc}")
            print("Smoke test: EXECUTION FAILED.")


if __name__ == "__main__":
    import resource
    main()
