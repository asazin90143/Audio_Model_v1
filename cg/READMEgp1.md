# Audio Processing Pipeline

This repository contains an end-to-end deep learning pipeline for environmental audio analysis, separation, and enhancement. The system supports both mono and multi-channel audio inputs.

## Features
- Adaptive input handling (mono or multi-channel)
- Denoising via a lightweight DAE
- Multi-channel source localization (GCC-PHAT)
- Audio event detection using a CRNN-based SED module
- Adaptive source separation (beamforming for multi-channel, mask-based for mono)
- Class-specific audio enhancement (speech enhancer, transient restorer)
- Generates separated and enhanced audio files plus a JSON report

## Directory Structure
- `cg/cg1.py`: Main pipeline implementation (Standalone Prototype)
- `out/`: Default output directory containing extracted sources, enhanced files, and the final report

## Installation
```bash
pip install numpy scipy soundfile librosa torch torchaudio
```

## 🚀 Usage & Testing

### 🧪 Test Run
To run the prototype on an input file:

```bash
python cg/cg1.py --input path/to/your/input.wav --outdir out/
```

### Command Line Arguments
```bash
python cg/cg1.py --input input.wav --outdir out/ --device cuda
```

### Optional Arguments
- `--model-dir`: Directory containing pretrained model checkpoints
- `--device`: `cpu` or `cuda`

## Output
Generated files include:
- Separated WAV files (beamformed or mask-separated)
- Enhanced WAV files (class-specific processing)
- `report.json` summarizing events, DOA, and output file paths

## Customization
Replace placeholder models with trained versions:
- Denoiser: Swap in a WaveNet or DAE
- Event Detector: Replace CRNN with AST or a large SED model
- Mono Separation: Replace MaskUNet with Conv-TasNet or SepFormer
- Multi-channel: Integrate neural beamformers with SELD DOA outputs

## License
MIT License

## Contact
For enhancement, improvements, or integration support, feel free to reach out.

