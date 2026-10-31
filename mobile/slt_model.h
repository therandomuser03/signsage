#ifndef SLT_MODEL_H_
#define SLT_MODEL_H_

#include <torch/script.h>
#include <string>
#include <vector>

struct CacheState {
    c10::intrusive_ptr<c10::ivalue::Tuple> sa_keys;
    c10::intrusive_ptr<c10::ivalue::Tuple> sa_vals;
    c10::intrusive_ptr<c10::ivalue::Tuple> ca_keys;
    c10::intrusive_ptr<c10::ivalue::Tuple> ca_vals;
};

class SLTModel {
public:
    explicit SLTModel(const std::string& model_dir);
    ~SLTModel() = default;

    std::string translate(const float* keypoints, int num_frames);

private:
    torch::jit::Module encoder_;
    torch::jit::Module t5_encoder_;
    torch::jit::Module embed_;
    torch::jit::Module first_step_;
    torch::jit::Module step_;
    torch::jit::Module lm_head_;

    torch::Device device_;
    int pad_id_ = 0;
    int eos_id_ = 1;
    int max_length_ = 128;
    int num_layers_ = 6;
    int num_heads_ = 8;
    int d_kv_ = 64;

    CacheState make_empty_cache();
    CacheState extract_cache(const c10::intrusive_ptr<c10::ivalue::Tuple>& out, int offset);
    torch::Tensor process_keypoints(const float* keypoints, int num_frames);
    torch::Tensor greedy_decode(const torch::Tensor& t5_enc_out);
};

#endif
