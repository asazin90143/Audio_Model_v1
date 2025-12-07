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