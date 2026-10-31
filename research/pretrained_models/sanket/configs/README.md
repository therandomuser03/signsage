---
license: mit
tags:
- vision
- sign-language-recognition
- action-recognition
- timesformer
- t5
- video-to-text
datasets:
- include
metrics:
- bleu
---

# Sanket - ISL Translator

This repository contains the trained weights and checkpoints for **Sanket - ISL Translator**, an end-to-end deep learning system designed for translating Indian Sign Language (ISL) videos into fluent English text. 

This model was developed as part of a Final Year Project and achieves a peak BLEU-1 score of **50.6%** on the validation set.

## Model Details

* **Architecture:** Dual-Stream Gated Multi-modal Transformer
* **Vision Encoder:** `facebook/timesformer-base-finetuned-k400` (Lower 8 blocks frozen)
* **Pose/Skeleton Encoder:** Custom 2-layer TransformerEncoder (processes 195 MediaPipe features/frame)
* **Language Decoder:** `t5-base` (220M parameters)
* **Dataset:** [INCLUDE Dataset](https://www.kaggle.com/datasets/daskoushik/include/data) (~4,284 videos across 250 Indian Sign Language categories)

## Training Configuration

The model was trained on an AWS `ml.g5.xlarge` instance (NVIDIA A10G 24GB VRAM) over 100 total epochs using a **Two-Phase Training Strategy**:

1. **Phase 1 (Epochs 1-20):** The `t5-base` language decoder was entirely frozen. Only the TimeSformer, PoseEncoder, and GatedCrossModalFusion layers were trained to produce stable visual embeddings.
2. **Phase 2 (Epochs 21-100):** The model was trained end-to-end. The T5 decoder was unfrozen, allowing cross-attention layers to adapt to the visual embeddings, mapping them to tokenized English text.

## Files Included

* **`model_best.pth`**: The best-performing model checkpoint based on Validation Loss. **Download this file** if you only want to run the Streamlit inference demo.
* **`checkpoint_latest.pth`**: The final training checkpoint containing the model weights, optimizer state, and scheduler state. Download this if you wish to resume or fine-tune training.

## How to Use with the GitHub Project

These weights are designed to act as the backbone for the main GitHub repository of this project.

1. Clone the Main GitHub Repository (e.g., `https://github.com/sanket/Sanket-ISL-Translator.git`).
2. Download the `model_best.pth` file from this Hugging Face page.
3. Place `model_best.pth` inside the `Sanket - ISL Translator/weights/` folder of your cloned repository.
4. Run the demo application:
```bash
cd "Sanket - ISL Translator"
pip install -r requirements.txt
streamlit run app.py
