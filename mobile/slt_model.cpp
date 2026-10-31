#include "slt_model.h"
#include <iostream>
#include <sstream>
#include <cassert>

static torch::TensorOptions long_opts(torch::Device device) {
    return torch::TensorOptions(device).dtype(torch::kLong);
}
static torch::TensorOptions float_opts(torch::Device device) {
    return torch::TensorOptions(device).dtype(torch::kFloat32);
}

SLTModel::SLTModel(const std::string& model_dir) {
    device_ = torch::kCPU;
    if (torch::cuda::is_available()) device_ = torch::kCUDA;

    encoder_       = torch::jit::load(model_dir + "/encoder_traced.pt", device_);
    t5_encoder_    = torch::jit::load(model_dir + "/t5_encoder.pt", device_);
    embed_         = torch::jit::load(model_dir + "/t5_shared_embed.pt", device_);
    first_step_    = torch::jit::load(model_dir + "/t5_decoder_first_step.pt", device_);
    step_          = torch::jit::load(model_dir + "/t5_decoder_step.pt", device_);
    lm_head_       = torch::jit::load(model_dir + "/t5_lm_head.pt", device_);

    encoder_.eval();
    t5_encoder_.eval();
    embed_.eval();
    first_step_.eval();
    step_.eval();
    lm_head_.eval();
}

CacheState SLTModel::make_empty_cache() {
    CacheState c;
    std::vector<torch::jit::IValue> keys, vals;
    for (int i = 0; i < num_layers_; i++) {
        auto empty = torch::empty({1, num_heads_, 0, d_kv_}, float_opts(device_));
        keys.push_back(empty);
        vals.push_back(empty);
    }
    c.sa_keys = c10::ivalue::Tuple::create(keys);
    c.sa_vals = c10::ivalue::Tuple::create(vals);
    c.ca_keys = c10::ivalue::Tuple::create(keys);  // clone fresh
    c.ca_vals = c10::ivalue::Tuple::create(vals);
    return c;
}

CacheState SLTModel::extract_cache(
    const c10::intrusive_ptr<c10::ivalue::Tuple>& out, int offset) {
    CacheState c;
    c.sa_keys = out->elements()[offset + 0].toTuple();
    c.sa_vals = out->elements()[offset + 1].toTuple();
    c.ca_keys = out->elements()[offset + 2].toTuple();
    c.ca_vals = out->elements()[offset + 3].toTuple();
    return c;
}

torch::Tensor SLTModel::process_keypoints(const float* keypoints, int num_frames) {
    TORCH_CHECK(num_frames <= 128, "num_frames must be <= 128, got ", num_frames);

    // (4, 94, T) → pad/crop to (4, 94, 128) then add batch dim
    auto inp = torch::from_blob((void*)keypoints, {4, 94, num_frames}, float_opts(torch::kCPU));
    if (num_frames < 128) {
        auto pad = torch::zeros({4, 94, 128 - num_frames}, float_opts(torch::kCPU));
        inp = torch::cat({inp, pad}, 2);
    } else if (num_frames > 128) {
        inp = inp.index({"...", torch::indexing::Slice(0, 128)});
    }
    inp = inp.unsqueeze(0);  // (1, 4, 94, 128)
    inp = inp.to(device_);

    // Step 1: visual encoder
    auto vis_out = encoder_.forward({inp}).toTensor();  // (1, 128, 512)

    // Step 2: T5 encoder
    auto t5_out = t5_encoder_.forward({vis_out}).toTensor();  // (1, 128, 512)
    return t5_out;
}

torch::Tensor SLTModel::greedy_decode(const torch::Tensor& t5_enc_out) {
    auto device = t5_enc_out.device();

    // Initialise with pad token
    torch::Tensor input_ids = torch::full({1, 1}, pad_id_, long_opts(device));
    torch::Tensor all_ids = input_ids.clone();

    // --- First decoder step (no cache → get cache from first_step) ---
    std::vector<torch::jit::IValue> first_inputs = {t5_enc_out, input_ids};
    auto first_out = first_step_.forward(first_inputs).toTuple();

    // first_out = (hidden, sa_keys, sa_vals, ca_keys, ca_vals)
    auto hidden = first_out->elements()[0].toTensor();  // (1, 1, 512)
    auto cache = extract_cache(first_out, 1);

    // Apply lm_head
    auto logits = lm_head_.forward({hidden}).toTensor();  // (1, 1, 32128)
    auto next_token = logits.argmax(-1);                  // (1, 1)
    all_ids = torch::cat({all_ids, next_token}, 1);
    input_ids = next_token;

    if (next_token.item<int>() == eos_id_) return all_ids;

    // --- Subsequent decoder steps with cache ---
    for (int i = 1; i < max_length_; i++) {
        std::vector<torch::jit::IValue> step_inputs = {
            t5_enc_out, input_ids,
            cache.sa_keys, cache.sa_vals, cache.ca_keys, cache.ca_vals
        };
        auto out = step_.forward(step_inputs).toTuple();

        // out = (logits, sa_keys, sa_vals, ca_keys, ca_vals)
        logits = out->elements()[0].toTensor();
        cache = extract_cache(out, 1);

        next_token = logits.argmax(-1);
        all_ids = torch::cat({all_ids, next_token}, 1);
        input_ids = next_token;

        if (next_token.item<int>() == eos_id_) break;
    }

    return all_ids;
}

std::string SLTModel::translate(const float* keypoints, int num_frames) {
    torch::NoGradGuard no_grad;

    auto t5_enc_out = process_keypoints(keypoints, num_frames);
    auto token_ids = greedy_decode(t5_enc_out);

    // TODO: decode token IDs to string using a loaded tokenizer on Java/Kotlin side
    // Return comma-separated IDs as a simple transport format
    std::ostringstream oss;
    for (int i = 0; i < token_ids.size(1); i++) {
        if (i > 0) oss << ",";
        oss << token_ids[0][i].item<int>();
    }
    return oss.str();
}
