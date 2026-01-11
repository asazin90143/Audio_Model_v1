# 🎧 Audio_Model_v1: Environmental Audio Analysis Project

Welcome to the **Audio_Model_v1** project! This repository hosts three distinct implementations of an end-to-end deep learning pipeline for environmental audio analysis. The system is designed to perform:
1.  **Denoising**
2.  **Source Localization** (Multi-channel)
3.  **Event Classification**
4.  **Source Separation**
5.  **Class-Specific Enhancement**

## 📂 Project Structure & Implementations

This project is divided into three main variations, each exploring a different architectural approach or implementation style:

### 1. 🐍 Core Deep Learning Pipeline (`c/`)
*   **Path**: `c/cl1.py` (Core Logic) & `cp/cp1.py` (Wrapper/CLI)
*   **Description**: The primary, fully-features PyTorch implementation. It includes specific neural network architectures for each stage (U-Net DAE, SELDNet, CRNN, Conv-TasNet/Beamformer, GANs).
*   **Best For**: Deep learning research, production-grade modeling, and full GPU acceleration.
*   **Documentation**: [Read more in c/READMEc1.md](c/READMEc1.md)

### 2. 🧩 Modular "EcoSense" System (`g/`)
*   **Path**: `g/g1.py`
*   **Description**: A highly modular, class-based design focusing on system architecture and extensibility. It defines clear interfaces for each stage, making it easy to swap out components (e.g., replacing a dummy localizer with a real one).
*   **Best For**: System design, architecture planning, and educational purposes.
*   **Documentation**: [Read more in g/READMEg1.md](g/READMEg1.md)

### 3. 🧪 Hybrid/Numpy Prototype (`cg/`)
*   **Path**: `cg/cg1.py`
*   **Description**: A self-contained prototype mixing PyTorch with `numpy`, `scipy`, and `librosa`. It implements lightweight versions of the models (e.g., a small "SmallCRNN", GCC-PHAT for localization) in a single script.
*   **Best For**: Rapid prototyping, understanding signal processing fundamentals, and running on CPU.
*   **Documentation**: [Read more in cg/READMEgp1.md](cg/READMEgp1.md)

---

## 🚀 Quick Start (Testing)

To quickly test each model, navigate to the root directory and run the following commands:

### Test Model C (Core Pipeline)
This runs the convenience wrapper `cp1.py`:
```bash
python cp/cp1.py --input path/to/your/audio.wav --outdir output_c
```
*Or run the synthetic test inside the core file:*
```bash
python c/cl1.py
```

### Test Model G (Modular System)
This runs the modular system simulation:
```bash
python g/g1.py
```

### Test Model CG (Prototype)
This runs the hybrid prototype:
```bash
python cg/cg1.py --input path/to/your/audio.wav --outdir output_cg
```

## 🛠️ Requirements

Ensure you have the necessary dependencies installed for all models:
```bash
pip install torch torchaudio numpy scipy librosa soundfile
```

---
*Created for Audio Forensic Analysis.*
