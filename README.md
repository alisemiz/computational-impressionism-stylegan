# Computational Impressionism: High-Fidelity Art Synthesis via Transfer Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1-ee4c2c.svg)](https://pytorch.org/)
[![StyleGAN2-ADA](https://img.shields.io/badge/StyleGAN2-ADA-green.svg)](https://github.com/NVlabs/stylegan2-ada-pytorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Synthesizing high-quality Impressionist artworks in resource-constrained environments using StyleGAN2-ADA and Transfer Learning.**

## 🎨 Abstract
This project introduces a methodological framework for "Computational Impressionism," aimed at synthesizing high-quality Impressionist artworks within a resource-constrained cloud environment (Google Colab via NVIDIA T4/A100 GPUs). By leveraging **StyleGAN2-ADA** and a strategic **Transfer Learning** approach initialized from the FFHQ manifold, we successfully mitigated data scarcity and achieved competitive FID scores.

## 🚀 Key Features
* **Resource-Constrained Optimization:** Optimized for training on accessible cloud GPUs (NVIDIA T4 16GB).
* **Transfer Learning Strategy:** Utilizing pre-trained FFHQ (human face) weights to accelerate convergence on artistic datasets.
* **High-Fidelity Generation:** Achieving **512x512** resolution with realistic brushstroke textures.
* **Automated Data Curation:** Custom pipeline for de-duplication using MD5 checksums.

## 📊 Experimental Results

We conducted a comparative analysis between rapid prototyping on NVIDIA T4 GPUs and rigorous validation on NVIDIA A100 GPUs.

### Quantitative Metrics
The model achieved its best performance at **512x512 resolution** with an **FID score of 57.66**.

| Resolution | Hardware | Protocol | FID ($\downarrow$) | KID ($\downarrow$) | Precision ($\uparrow$) | Recall ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **64x64** | T4 | Turbo (5k) | 73.21 | 0.033 | 0.642 | 0.198 |
| **128x128** | T4 | Turbo (5k) | **59.64** | 0.017 | 0.698 | 0.123 |
| **256x256** | T4 | Turbo (5k) | 67.36 | 0.025 | 0.611 | 0.146 |
| **512x512** | **A100** | **Full (50k)** | **57.66** | **0.012** | 0.625 | **0.234** |

### Qualitative Samples
*Generated samples demonstrating the "impasto" technique and lighting composition.*

<p align="center">
  <img src="results/512_sample1.png" width="30%" />
  <img src="results/512_sample2.png" width="30%" />
  <img src="results/512_sample3.png" width="30%" />
</p>

## 🛠️ Installation & Usage

This project is designed to run on **Google Colab**.

1.  **Clone the repository:**
    ```bash
    git clone (https://github.com/alisemiz/computational-impressionism.git)
    cd computational-impressionism
    ```

2.  **Install Dependencies:**
    ```bash
    pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
    pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
    ```

3.  **Training (Example Command):**
    ```bash
    python train.py --outdir=./results --data=./datasets/impressionism.zip \
    --gpus=1 --cfg=auto --kimg=2500 --aug=bgcfnc --target=0.6 \
    --resume=ffhq1024 --snap=50
    ```

## 📂 Dataset
The dataset was curated from open-source repositories (WikiArt, Wikimedia Commons), focusing strictly on the **Impressionism** movement. 
* **Preprocessing:** Lanczos resampling.
* **Cleaning:** Automated duplicate removal via cryptographic hashing.

## 👥 Authors
* **Ali Semiz** - [GitHub Profile](https://github.com/alisemiz)
* **Ferhat Bozkurt**

## 📄 Citation
If you use this work, please cite our paper:

```bibtex
@inproceedings{semiz2025computational,
  title={Computational Impressionism: High-Fidelity Art Synthesis via Transfer Learning in Resource-Constrained Environments},
  author={Semiz, Ali and Bozkurt, Ferhat},
  year={2025},
  institution={Ataturk University}
}
