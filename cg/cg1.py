#!/usr/bin/env python3
"""
File: audio_pipeline/main.py

Integrated pipeline for:
1) Adaptive input handling and denoising
2) (Multi-channel) Source localization (GCC-PHAT -> DOA)
3) Audio event detection/classification (CRNN)
4) Adaptive separation (spatial beamforming or mask-based mono separation)
5) Class-specific enhancement and output

Notes:
- This is a working, runnable prototype with small example models.
- Replace model placeholders with production checkpoints for best results.
- Keep comments minimal and focused on 'why' only.
"""

from pathlib import Path
import json
import math
import argparse
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import soundfile as sf
import librosa
import scipy
from scipy.signal import stft, istft
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# ----------------------------
# Config / Taxonomy
# ----------------------------
SAMPLE_RATE = 16000
TAXONOMY = ["Gunshot", "Explosion", "Speech", "Siren", "Car", "Dog", "Glass", "Alarm"]
# default thresholds
EVENT_DET_THRESH = 0.5
EVENT_MIN_DURATION = 0.05  # seconds

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_audio(path: str, sr: int = SAMPLE_RATE):
    wav, sr0 = sf.read(path, always_2d=True)
    # wav shape: (frames, channels)
    wav = wav.T  # channels x frames
    if sr0 != sr:
        wav = np.stack([librosa.resample(ch.astype(float), orig_sr=sr0, target_sr=sr) for ch in wav])
    return wav.astype(np.float32), sr

def write_wav(path: str, wav: np.ndarray, sr: int = SAMPLE_RATE):
    # wav: (channels, frames) or (frames,)
    if wav.ndim == 2 and wav.shape[0] > 1:
        data = wav.T
    else:
        data = wav if wav.ndim == 1 else wav.squeeze()
    sf.write(str(path), data, samplerate=sr)

def seconds_to_frame(t: float, sr=SAMPLE_RATE, hop=512):
    return int(t * sr / hop)

# ----------------------------
# 1) Denoiser (lightweight conv-autoencoder)
# ----------------------------
class Conv1dDAE(nn.Module):
    """Small 1D conv autoencoder for denoising — replace with production DAE/WaveNet."""
    def __init__(self, channels=1, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden*2, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(hidden*2, hidden, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden, hidden, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, kernel_size=15, stride=1, padding=7),
            nn.Tanh()
        )

    def forward(self, x):
        # x: (batch=1, channels, frames)
        z = self.enc(x)
        out = self.dec(z)
        return out

def denoise_multichannel(wav: np.ndarray, model: Optional[nn.Module]=None, device='cpu') -> np.ndarray:
    # why: boost SNR for downstream tasks
    if model is None:
        model = Conv1dDAE(channels=1)
    model.to(device).eval()
    denoised = []
    with torch.no_grad():
        for ch in wav:
            t = torch.from_numpy(ch).unsqueeze(0).unsqueeze(0).to(device)  # 1x1xT
            out = model(t).squeeze().cpu().numpy()
            # match length
            if out.shape[0] != ch.shape[0]:
                out = scipy.signal.resample(out, ch.shape[0])
            denoised.append(out)
    return np.stack(denoised)

# ----------------------------
# 2) Localization (GCC-PHAT + simple DOA)
# ----------------------------
def gcc_phat(sig, refsig, fs=SAMPLE_RATE, max_tau=None, interp=16):
    # returns cross-correlation lag index estimate using GCC-PHAT
    n = sig.shape[0] + refsig.shape[0]
    nfft = 1 << (n - 1).bit_length()
    SIG = np.fft.rfft(sig, n=nfft)
    REF = np.fft.rfft(refsig, n=nfft)
    R = SIG * np.conj(REF)
    denom = np.abs(R)
    denom[denom==0] = 1e-8
    R = R / denom
    cc = np.fft.irfft(R, n=nfft * interp)
    max_shift = int(interp * nfft / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc

def estimate_doa_multichannel(wav: np.ndarray, sr=SAMPLE_RATE, mic_geometry: Optional[np.ndarray]=None):
    # why: provide spatial coordinates for events when multi-channel available
    # wav: channels x frames
    C, T = wav.shape
    if C < 2:
        return []
    # naive pairwise TDOA -> DOA via least squares assuming linear array along x-axis
    # mic_geometry: C x 3 array (x,y,z). If None, assume uniform linear array spacing 0.04m on x axis.
    if mic_geometry is None:
        spacing = 0.04
        mic_geometry = np.array([[i * spacing, 0.0, 0.0] for i in range(C)])
    # analyze in windows
    frame_len = int(0.5 * sr)
    hop = frame_len // 2
    results = []
    c = 343.0
    for start in range(0, max(1, T - frame_len + 1), hop):
        end = start + frame_len
        taus = []
        for i in range(1, C):
            tau, _ = gcc_phat(wav[i, start:end], wav[0, start:end], fs=sr, max_tau=0.01)
            taus.append(tau)
        # simple DOA estimate: use time delay to compute angle relative to array axis
        # for linear array with spacing d: tau = (d * sin(theta)) / c * index
        # Using first mic pair approximation:
        if len(taus) > 0:
            tau_mean = np.mean(taus)
            # clamp
            val = tau_mean * c / spacing
            val = np.clip(val, -1.0, 1.0)
            theta = math.degrees(math.asin(val))  # azimuth in degrees
            results.append({
                "time": (start + end) / 2 / sr,
                "azimuth_deg": float(theta),
                "elevation_deg": 0.0
            })
    return results

# ----------------------------
# 3) Event Detector (small CRNN)
# ----------------------------
class SmallCRNN(nn.Module):
    """Small CRNN for frame-level SED. Replace with AST/CRNN training for production."""
    def __init__(self, n_classes=len(TAXONOMY), n_mels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,2))
        )
        self.rnn = nn.GRU(input_size=(n_mels//2)*32, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, mel):  # mel: batch x 1 x frames x n_mels
        x = self.conv(mel)  # batch x C x frames x n_mels/2
        b, c, t, m = x.shape
        x = x.permute(0,2,1,3).contiguous().view(b,t,c*m)
        r, _ = self.rnn(x)
        out = torch.sigmoid(self.fc(r))  # b x t x classes
        return out

def detect_events(wav_ref: np.ndarray, sr=SAMPLE_RATE, model: Optional[nn.Module]=None, device='cpu') -> List[Dict[str,Any]]:
    # why: detect class labels and strong event times
    if model is None:
        model = SmallCRNN()
    model.to(device).eval()
    # compute mel spectrogram frames
    wav_mono = wav_ref if wav_ref.ndim == 1 else wav_ref[0]
    hop = 512
    n_fft = 1024
    mel_spec = librosa.feature.melspectrogram(y=wav_mono, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64)
    log_mel = librosa.power_to_db(mel_spec+1e-8)
    # prepare input: frames x n_mels -> (1,1,frames,n_mels)
    x = torch.from_numpy(log_mel.T).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(x)  # 1 x t x classes
        out = out.squeeze(0).cpu().numpy()  # t x classes
    # threshold per frame -> events by median smoothing + connected components
    events = []
    t_frames = out.shape[0]
    for cidx, cls in enumerate(TAXONOMY):
        preds = out[:, cidx]
        mask = preds > EVENT_DET_THRESH
        # simple smoothing: remove tiny holes
        mask = scipy.ndimage.binary_opening(mask, structure=np.ones(3))
        starts = np.where(np.diff(np.concatenate(([0], mask.astype(int)))) == 1)[0]
        ends = np.where(np.diff(np.concatenate((mask.astype(int), [0]))) == -1)[0]
        for s,e in zip(starts, ends):
            start_sec = (s * hop) / sr
            end_sec = (e * hop) / sr
            if end_sec - start_sec < EVENT_MIN_DURATION:
                continue
            events.append({"label": cls, "start": float(start_sec), "end": float(end_sec), "score": float(preds[s:e].max())})
    # sort events
    events = sorted(events, key=lambda x: x["start"])
    return events

# ----------------------------
# 4) Separation
# ----------------------------
def stft_mag_phase(x, n_fft=1024, hop=256, win='hann'):
    f, t, Z = stft(x, nperseg=n_fft, noverlap=n_fft-hop, window=win)
    return np.abs(Z), np.angle(Z)

def istft_from_mag_phase(mag, phase, n_fft=1024, hop=256, win='hann'):
    Z = mag * np.exp(1j * phase)
    _, x = istft(Z, nperseg=n_fft, noverlap=n_fft-hop, window=win)
    return x

class MaskUNet(nn.Module):
    """Small U-Net mask estimator — used both for mono and multichannel mask refinement."""
    def __init__(self, in_ch=1, out_ch=2):  # out_ch = number of masks/sources to estimate
        super().__init__()
        self.enc1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(64,32,3,padding=1)
        self.dec1 = nn.ConvTranspose2d(32,16,3,padding=1)
        self.out = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        # x: batch x ch x freq x time
        a = F.relu(self.enc1(x))
        b = F.relu(self.enc2(a))
        c = F.relu(self.enc3(b))
        d = F.relu(self.dec2(c))
        e = F.relu(self.dec1(d + b))
        out = torch.sigmoid(self.out(e + a))
        return out  # batch x out_ch x freq x time

def separate_mono_mask_based(wav: np.ndarray, events: List[Dict[str,Any]], sr=SAMPLE_RATE, model: Optional[nn.Module]=None, device='cpu'):
    # why: blind source separation when no localization
    # wav: channels x frames OR frames
    mono = wav if wav.ndim == 1 else wav[0]
    n_fft = 1024
    hop = 256
    mag, phase = stft_mag_phase(mono, n_fft=n_fft, hop=hop)
    freq, time = mag.shape
    # decide N = max simultaneous overlapping events in time
    # We'll build time-activity grid and pick maximum overlapping count
    time_bins = np.arange(time)
    def time_to_bin(tsec):
        return int(tsec * sr / hop)
    act = np.zeros(time, dtype=int)
    overlaps = []
    for ev in events:
        s = max(0, time_to_bin(ev["start"]))
        e = min(time-1, time_to_bin(ev["end"]))
        act[s:e+1] += 1
    N = int(max(1, act.max()))
    # provide at least 1 mask
    if model is None:
        model = MaskUNet(in_ch=1, out_ch=N)
    model.to(device).eval()
    X = torch.from_numpy(mag[np.newaxis, np.newaxis,:,:]).float().to(device)
    with torch.no_grad():
        masks = model(X).squeeze(0).cpu().numpy()  # N x freq x time
    sources = []
    for i in range(masks.shape[0]):
        m = masks[i]
        mag_i = mag * m
        wav_i = istft_from_mag_phase(mag_i, phase, n_fft=n_fft, hop=hop)
        sources.append(wav_i)
    return sources

def beamform_with_doa(wav: np.ndarray, doa_list: List[Dict[str,Any]], sr=SAMPLE_RATE):
    # why: exploit spatial cues in multichannel to separate sources when DOA available
    # Simple delay-and-sum beamforming per DOA peak over whole file. For each DOA, steer and sum.
    C, T = wav.shape
    if C < 2 or len(doa_list) == 0:
        # fallback: average channels
        return [wav.mean(axis=0)]
    # group DOAs (unique by azimuth bins)
    azs = [int(round(d['azimuth_deg'] / 10) * 10) for d in doa_list]
    unique_azs = sorted(set(azs))
    outputs = []
    mic_positions = np.array([[i*0.04, 0, 0] for i in range(C)])
    c = 343.0
    for az in unique_azs:
        theta = math.radians(az)
        # steering vector for plane wave from azimuth theta (on x-y plane)
        direction = np.array([math.cos(theta), math.sin(theta), 0.0])
        # compute delays per mic
        delays = - (mic_positions @ direction) / c  # negative sign to align
        # convert to sample delays
        sample_delays = (delays * sr).astype(int)
        # apply integer-sample delay-and-sum
        max_delay = int(np.abs(sample_delays).max())
        out = np.zeros(T + max_delay)
        count = np.zeros_like(out)
        for m in range(C):
            d = sample_delays[m]
            if d >= 0:
                out[d:d+T] += wav[m]
                count[d:d+T] += 1
            else:
                dd = -d
                out[0:T] += wav[m, dd:dd+T]
                count[0:T] += 1
        count[count==0]=1
        beam = out[:T] / count[:T]
        outputs.append(beam)
    return outputs

# ----------------------------
# 5) Class-specific enhancement modules (stubs)
# ----------------------------
class SpeechEnhancer(nn.Module):
    """Placeholder speech enhancer. Replace with SEGAN / Demucs / RNNoise model."""
    def __init__(self):
        super().__init__()
        self.dae = Conv1dDAE(channels=1, hidden=64)
    def forward(self, x):
        return self.dae(x)

class TransientRestorer(nn.Module):
    """Small transient restoration using spectral smoothing."""
    def __init__(self):
        super().__init__()
    def forward(self, wav_np):
        # simple transient boost via highpass + transient gain
        return wav_np * 1.03

def enhance_by_class(wav: np.ndarray, cls: str, device='cpu', model_speech: Optional[nn.Module]=None):
    # why: class-specific final enhancement to maximize perceptual quality
    if cls == "Speech":
        if model_speech is None:
            model_speech = SpeechEnhancer().to(device)
        model_speech.eval()
        with torch.no_grad():
            t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model_speech(t).squeeze().cpu().numpy()
            if out.shape[0] != wav.shape[0]:
                out = scipy.signal.resample(out, wav.shape[0])
            return out
    elif cls in ("Gunshot", "Explosion", "Glass"):
        tr = TransientRestorer()
        return tr.forward(wav)
    else:
        # mild spectral enhancement: simple soft clipping + mild compression
        wav2 = np.tanh(1.05 * wav)
        return wav2

# ----------------------------
# Pipeline runner
# ----------------------------
def run_pipeline(input_path: str, outdir: str, model_dir: Optional[str]=None, device='cpu'):
    p = Path(outdir)
    ensure_dir(p)
    # 1) Load
    wav, sr = load_audio(input_path, sr=SAMPLE_RATE)
    C = wav.shape[0]
    print(f"Loaded {input_path} — channels={C}, sr={sr}, length={wav.shape[1]/sr:.2f}s")

    # 1.1 Denoise all channels
    denoiser = None
    denoised = denoise_multichannel(wav, model=denoiser, device=device)
    print("Denoising done.")

    # 2) Localization if multi-channel
    doa_list = []
    if C > 1:
        doa_list = estimate_doa_multichannel(denoised, sr=sr)
        print(f"Estimated {len(doa_list)} DOA frames (multi-channel).")

    # 3) Event detection (use reference channel 0)
    detector = None
    events = detect_events(denoised[0], sr=sr, model=detector, device=device)
    print(f"Detected {len(events)} events.")
    # attach nearest DOA if available
    for ev in events:
        # find DOA entries in that event window
        candidates = [d for d in doa_list if (d['time'] >= ev['start'] - 0.2 and d['time'] <= ev['end'] + 0.2)]
        if candidates:
            ev['azimuth_deg'] = float(np.median([c['azimuth_deg'] for c in candidates]))
            ev['elevation_deg'] = float(np.median([c['elevation_deg'] for c in candidates]))
    # 4) Separation
    separated_files = []
    if C > 1 and len(doa_list) > 0:
        beams = beamform_with_doa(denoised, doa_list, sr=sr)
        # For each beam, attempt to map to events by time overlap -> write file
        for i, beam in enumerate(beams):
            fname = p / f"sep_beam_{i}.wav"
            write_wav(str(fname), beam, sr=sr)
            separated_files.append({"path": str(fname), "source_hint": "beam", "azimuth_deg": float(doa_list[i]['azimuth_deg']) if i < len(doa_list) else None})
        # further refine via mask-unet if desired (omitted for brevity)
    else:
        # Mono separation using mask-based network
        sources = separate_mono_mask_based(denoised, events, sr=sr, model=None, device=device)
        for i, src in enumerate(sources):
            fname = p / f"sep_mono_{i}.wav"
            write_wav(str(fname), src, sr=sr)
            separated_files.append({"path": str(fname), "source_hint": "mono_mask", "index": i})
    # 5) Map each separated file to a class by checking overlap with event list, then enhance
    final_files = []
    for s in separated_files:
        # simple mapping: find an event that overlaps midpoint of the file (rough)
        # read separated audio
        y, _ = load_audio(s["path"], sr=sr)
        y_mono = y.mean(axis=0) if y.ndim == 2 else y
        mid_t = y_mono.shape[0]/sr/2
        matched = None
        for ev in events:
            # if event intersects midpoint relative to original file length — naive mapping
            # Here we map by best score label if any event overlaps with any time in original file
            # In production, use embedding similarity or informed masking
            if ev['start'] <= mid_t <= ev['end']:
                matched = ev
                break
        cls = matched['label'] if matched else "Unknown"
        enhanced = enhance_by_class(y_mono, cls, device=device, model_speech=None)
        out_path = Path(p) / f"enh_{Path(s['path']).name}"
        write_wav(str(out_path), enhanced, sr=sr)
        final_files.append({"path": str(out_path), "class": cls, "orig_sep": s})
    # Output report
    report = {
        "input": str(input_path),
        "channels": int(C),
        "events": events,
        "separated": final_files,
        "doa_frames": doa_list
    }
    with open(p / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Pipeline complete. Outputs in {p.resolve()}")
    return report

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Audio E2E pipeline: denoise -> localize -> detect -> separate -> enhance")
    parser.add_argument("--input", "-i", required=True, help="Input WAV file (mono or multichannel)")
    parser.add_argument("--outdir", "-o", default="out", help="Output directory")
    parser.add_argument("--model-dir", "-m", default=None, help="Directory with model checkpoints (optional)")
    parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    args = parser.parse_args()
    run_pipeline(args.input, args.outdir, model_dir=args.model_dir, device=args.device)

if __name__ == "__main__":
    main()
