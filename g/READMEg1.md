# 🌲 EcoSense-Audio: End-to-End Environmental Audio Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![Status](https://img.shields.io/badge/Status-Development-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

**EcoSense-Audio** is a robust Deep Learning framework designed for the advanced analysis, separation, and enhancement of environmental soundscapes. It features a dynamic architecture capable of handling both **Single-Channel (Mono)** and **Multi-Channel (Microphone Array)** inputs.

The system processes raw audio through a sequential 5-stage pipeline to detect events (e.g., "Gunshot", "Speech"), localize them in 3D space (if supported), separate the audio sources, and apply class-specific enhancement.

---

## 🚀 Key Features

The pipeline consists of five modular blocks:

### 1. ⚙️ Adaptive Pre-processing & Denoising
* **Input Analysis:** Automatically detects channel count ($C$).
* **Signal Cleaning:** Uses a **Denoising Autoencoder (DAE)** to remove background hiss and improve Signal-to-Noise Ratio (SNR) before analysis.

### 2. 📍 Audio Source Localization (ASL)
* *Active only if $C > 1$ (Multi-channel).*
* **SELD Network:** Estimates Direction-of-Arrival (DOA) including Azimuth ($\phi$) and Elevation ($\theta$).
* **Mapping:** Associates spatial coordinates with specific timestamps.

### 3. 🏷️ Event Classification (SED)
* **Architecture:** Convolutional Recurrent Neural Network (CRNN) or Audio Spectrogram Transformer (AST).
* **Taxonomy:** Detects environmental classes: *Gunshot, Explosion, Speech, Siren, Car Engine, Dog Bark, Breaking Glass, Alarm.*
* **Output:** Precise start/end timestamps and class labels.

### 4. 🔗 Adaptive Source Separation
* **Mono Mode ($C=1$):** Utilizes **Blind Source Separation (BSS)** (e.g., Conv-TasNet) with Permutation Invariant Training (PIT).
* **Array Mode ($C>1$):** Combines localization data with **Mask-based Beamforming** for spatially-informed separation.

### 5. ✨ Class-Specific Enhancement
* **Targeted Refinement:** Applies specific models based on the detected class.
    * *Speech* $\rightarrow$ Speech Enhancement GAN.
    * *Gunshots/Glass* $\rightarrow$ Transient Restoration Model.
* **Result:** High-fidelity, isolated `.wav` files for every detected source.

---

## 📂 Project Structure

```text
EcoSense-Audio/
├── data/                   # Input audio files and raw datasets
├── models/                 # Pre-trained weights (.pth files)
├── modules/                # Core logic for the 5 stages
│   ├── __init__.py
│   ├── preprocessor.py     # Denoising and Input handling
│   ├── localizer.py        # SELD / DOA estimation
│   ├── classifier.py       # CRNN / SED logic
│   ├── separator.py        # Conv-TasNet / Beamforming
│   └── enhancer.py         # Post-processing GANs
├── utils/                  # Helper functions (audio I/O, spectrograms)
├── main.py                 # Pipeline orchestrator
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

# EcoSense Audio

## 📦 Installation

Follow these steps to set up the environment.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ecosense-audio.git
cd ecosense-audio
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

**MacOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install the required PyTorch and audio processing libraries.

```bash
pip install -r requirements.txt
```

**Note:** If you have a CUDA-enabled GPU, ensure you install the specific version of torch compatible with your CUDA drivers.

## 💻 Usage

### 🧪 Testing
This implementation is designed as a modular framework. You can run the file directly to verify the architecture initialization:

```bash
python g/g1.py
```
*Note: The default `__main__` block initializes the system but does not process a file by default to avoid errors. You can modify the bottom of `g/g1.py` to point to a real wav file.*

### Python API

You can import the core system into your own Python scripts:

```python
import torchaudio
from g.g1 import EnvironmentalAudioSystem

# 1. Initialize the system
system = EnvironmentalAudioSystem()

# 2. Process an audio file
# The system automatically detects if the input is Mono or Multi-channel
input_path = "data/forest_ambience.wav"
results = system.process_file(input_path)

# 3. Save the separated, enhanced outputs
for i, (label, waveform) in enumerate(results):
    output_filename = f"output_{i}_{label}.wav"
    torchaudio.save(output_filename, waveform, sample_rate=16000)
    print(f"Saved: {output_filename}")
```

### 🖥️ CLI Usage
Currently, this module is set up as a framework. To run it as a CLI, you would need to implement an `argparse` wrapper similar to the one in `c/cl1.py` or `cp/cp1.py`.

**Arguments:**
- `--input`: Path to the input `.wav` file.
- `--output_dir`: Directory where separated source files will be saved.
- `--threshold`: Confidence threshold for event detection (default: 0.5).
- `--save_spectrograms`: Flag to generate and save visual plots of the analysis.

## 🏗️ Technical Architecture

The system is built on a modular "Listen-Attend-Separate" paradigm.

### Data Flow Pipeline

1. **Input Ingestion:** Raw Audio Tensor $X \in \mathbb{R}^{C \times T}$.

2. **Denoising (Stage 1):** $X_{clean} = f_{DAE}(X)$
   - Model: 1D-Convolutional Autoencoder.

3. **Localization (Stage 2):** $L = f_{SELD}(X_{clean})$
   - Output: Coordinates $(\phi, \theta)$ per time-step. Active only if $C > 1$.

4. **Detection (Stage 3):** $E = f_{CRNN}(X_{clean})$
   - Output: List of tuples $(t_{start}, t_{end}, class_{label})$.

5. **Separation (Stage 4):**
   - If Mono: $S = f_{BSS}(X_{clean})$ (Blind Separation).
   - If Multi: $S = f_{Beam}(X_{clean}, L)$ (Spatial Separation).

6. **Enhancement (Stage 5):** $S_{final} = f_{ENH}(S, class_{label})$
   - Logic: Routes audio to specific GANs based on the classification label.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your Changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the Branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

Please ensure all new modules include appropriate unit tests in the `tests/` directory.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.