# Replicating Vision Transformer (ViT) for Image Recognition

## 📘 Project Overview
This repository documents the replication and practical application of the Vision Transformer (ViT) architecture from the seminal paper:

**“An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale”**  
*Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, et al., 2021*  
**arXiv: 2010.11929v2**

The goal of this project is to understand, implement, and apply the ViT model, demonstrating its effectiveness in real-world computer vision tasks via transfer learning.

The full workflow—including replication, training, and fine-tuning—is documented across three Jupyter notebooks:

- `Replicating_Paper.ipynb`
- `torch_CV_training.ipynb`
- `transfer_learning.ipynb`

---

## 📂 Project Structure

| Path / File | Description |
|-------------|-------------|
| `food_data/` | Image dataset used for transfer learning |
| `food_data/pizza_steak_sushi/` | The specific 3-class dataset used for fine-tuning |
| `models/` | Stores trained / fine-tuned model weights |
| `models/08_pretrained_vit_...pizza_steak_sushi.pt` | Final fine-tuned ViT model checkpoint |
| `Replicating_Paper.ipynb` | ViT implementation & replication of paper experiments |
| `torch_CV_training.ipynb` | Foundational PyTorch CV training (CNNs, loops, transforms) |
| `transfer_learning.ipynb` | ViT transfer learning: loading, replacing head, fine-tuning |

---

## 🧠 The Vision Transformer (ViT) Architecture

ViT adapts the standard Transformer encoder—originally for NLP—to image data by treating images as sequences of patches. Its core ideas:

### **1. Image Patching**
Images are split into fixed-size non-overlapping patches (e.g., **16×16**).  
Each patch becomes a “token” for the Transformer.

### **2. Linear Patch Embedding**
Each flattened patch is projected into a Transformer dimension \( D \), producing the embedding sequence.

### **3. [CLS] Token**
A learnable classification token is prepended.  
Its final output embedding is used for classification.

### **4. Position Embeddings**
1D trainable position embeddings preserve spatial information.

### **5. Transformer Encoder**
Sequence is processed by stacked encoder layers consisting of:
- Multi-Head Self-Attention (MSA)
- MLP blocks
- LayerNorm & residual connections

Despite having **no convolutional layers**, ViT achieves SOTA when trained on large datasets.

---

## 🏋️‍♂️ Training & Transfer Learning Workflow

### **1. Foundational Computer Vision Training**  
Notebook: `torch_CV_training.ipynb`

Topics covered:
- Dataset loading & augmentation (`torchvision.datasets`, `transforms`)
- Building and training CNN baselines
- Core PyTorch training loop (loss, optimizer, evaluation)

---

### **2. Vision Transformer Transfer Learning**  
Notebook: `transfer_learning.ipynb`

Steps performed:
- **Load a pre-trained ViT** (ImageNet-21k / JFT-300M)
- **Replace classifier head** for 3-class task: *pizza / steak / sushi*
- **Fine-tune** the entire model or head-only
- **Save final checkpoint** to: **models/08_pretrained_vit_…pizza_steak_sushi.pt**


This yields a high-accuracy classifier despite limited data—thanks to ViT’s powerful pretraining.

---

## 📚 References

1. **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**  
   Dosovitskiy et al., 2021  
   https://arxiv.org/abs/2010.11929v2

2. **Vision Transformer: What It Is & How It Works (2024 Guide)**  
   V7 Labs  
   https://www.v7labs.com/blog/vision-transformer-guide

---
