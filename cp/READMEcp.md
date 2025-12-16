# 🎧 Environmental Audio Analysis, Separation, and Enhancement Pipeline

## Overview
This project implements a **deep learning-based end-to-end system** for robust analysis, separation, and enhancement of environmental audio.  
It supports both **single-channel (mono)** and **multi-channel (microphone array)** inputs, dynamically adapting its processing pipeline based on input configuration.

The system outputs:
- A **list of classified audio events** with timestamps (and spatial coordinates if multi-channel).
- A set of **individually separated and enhanced audio files (WAV)** corresponding to each detected source.

---

## 🔑 Key Features
- **Adaptive Input Handling**: Automatically detects channel count and configures pipeline accordingly.
- **Deep Denoising**: Noise reduction via DAE or WaveNet-based models.
- **Source Localization (Multi-channel)**: SELD network estimates DOA (azimuth, elevation).
- **Event Classification & Detection**: CRNN or Transformer-based architectures for strong event detection.
- **Adaptive Source Separation**:
  - Multi-channel: Spatially-informed beamforming + mask networks.
  - Mono: Blind source separation (Conv-TasNet / SepFormer).
- **Class-Specific Enhancement**: Modular enhancement tailored to event type (speech, gunshot, siren, etc.).

---

## 📦 Pipeline Architecture

### 1. ⚙️ Adaptive Input & Pre-processing
- Detects channel count (C).
- Denoises audio using DAE/WaveNet.
- Normalizes and resamples input.

### 2. 📍 Audio Source Localization (C > 1 only)
- SELD network trained on microphone array data.
- Outputs event timestamps, labels, and DOA (ϕ, θ).

### 3. 🏷️ Event Classification & Detection
- CRNN or Transformer-based classifier.
- Outputs `(Class Label, Start Time, End Time)`.

### 4. 🔗 Adaptive Source Separation
- **Multi-channel**: Beamforming + DOA-informed mask networks.
- **Mono**: Blind source separation (Conv-TasNet / SepFormer).
- Outputs N separated waveform files.

### 5. ✨ Class-Specific Enhancement
- Speech → SEGAN/DNS-style enhancement.
- Gunshot/Explosion → Transient restoration.
- Siren/Alarm → Harmonic enhancement.
- Car engine → Low-frequency stabilization.

---

## 🗂️ Output
- **Event List**: JSON or CSV containing detected events.
- **Separated Audio**: WAV files named by event type and timestamp.

Example:
- events.json ├── [ { "class": "Speech", "start": 1.2, "end": 4.5 } ] outputs/ ├── 0001_speech.wav ├── 0002_siren.wav


---

## ⚙️ Installation

### Requirements
- Python 3.9+
- PyTorch / TensorFlow (depending on chosen models)
- Librosa, NumPy, SciPy
- Soundfile, PyTorch Lightning (recommended)

### Setup
```bash
git clone https://github.com/your-repo/audio-pipeline.git
cd audio-pipeline
pip install -r requirements.txt


### 🚀 Usage
Run pipeline on an audio file
python run_pipeline.py --input input.wav --output_dir outputs/


Options
- --array_geometry: JSON file with microphone positions (required for multi-channel).
- --sample_rate: Target sample rate (default: 24000).
- --model_config: Path to YAML config for model selection.

📊 Evaluation Metrics
- Denoising: SI-SDR, SI-SNR.
- Localization: DOA error (degrees).
- Classification: Event-based F1, segment-based F1.
- Separation: SI-SDR, SDR, SIR, SAR.
- Enhancement: PESQ, STOI, transient sharpness indices.

🛠️ Project Structure
audio-pipeline/
├── configs/              # YAML configs for models
├── checkpoints/          # Pretrained weights
├── data/                 # Sample datasets
├── modules/              # Core pipeline modules
│   ├── preprocessing.py
│   ├── localization.py
│   ├── classification.py
│   ├── separation.py
│   └── enhancement.py
├── run_pipeline.py       # Main entry point
└── README.md



📚 Training Data
- Multi-channel: DCASE SELD datasets.
- Mono: ESC-50, UrbanSound8K, LibriSpeech, DNS Challenge.
- Synthetic mixtures: Augmented with impulse responses and noise.

🤝 Contributing
Contributions are welcome!
- Fork the repo
- Create a feature branch
- Submit a pull request

📜 License
MIT License. See LICENSE file for details.

🌟 Future Work
- Real-time inference optimization.
- Expanded taxonomy of environmental sounds.
- Unified enhancement model for cross-class generalization.

---

Would you like me to also draft a **minimal repo skeleton with `run_pipeline.py` and configs** so you can drop this README in and have a working starter structure? That way, you’d have both documentation and code scaffolding ready to go.





## to run

# 1st

# (optional) activate your venv if you use it
# .\g\venv\Scripts\Activate.ps1

# install CPU wheel
# python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

## Then rerun:
# (python cp\cp1.py --input path\to\file.wav)