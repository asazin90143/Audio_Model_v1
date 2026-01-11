"""
End-to-End Deep Learning System for Environmental Audio Analysis
Implements: Denoising → Localization → Classification → Separation → Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings
import soundfile as sf
warnings.filterwarnings('ignore')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AudioEvent:
    """Represents a detected audio event"""
    label: str
    start_time: float
    end_time: float
    confidence: float
    azimuth: Optional[float] = None  # ϕ (degrees)
    elevation: Optional[float] = None  # θ (degrees)


@dataclass
class SeparatedSource:
    """Represents a separated audio source"""
    waveform: torch.Tensor
    sample_rate: int
    event: AudioEvent
    enhanced: bool = False


# ============================================================================
# STAGE 1: ADAPTIVE INPUT AND PRE-PROCESSING
# ============================================================================

class DenoisingAutoencoder(nn.Module):
    """
    Deep Neural Network-based Denoising Autoencoder (DAE)
    Uses U-Net architecture for spectral refinement
    """
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder with skip connections (U-Net style)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input spectrogram [B, 1, F, T]
        Returns:
            Denoised spectrogram mask [B, 1, F, T]
        """
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder with skip connections
        d3 = self.dec3(b)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        
        # Output mask
        mask = self.final(d1)
        return mask


class AdaptivePreprocessor:
    """Stage 1: Adaptive Input and Pre-processing Layer"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.denoiser = DenoisingAutoencoder().to(device)
        self.denoiser.eval()
        
        self.n_fft = 2048
        self.hop_length = 512
        self.window = torch.hann_window(self.n_fft).to(device)
    
    def process(self, audio: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int]:
        """
        Denoise audio across all channels
        
        Args:
            audio: [C, T] tensor
            sr: Sample rate
            
        Returns:
            Denoised audio [C, T], channel count
        """
        C = audio.shape[0]
        denoised_channels = []
        
        with torch.no_grad():
            for c in range(C):
                # STFT
                spec = torch.stft(
                    audio[c].to(self.device),
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window,
                    return_complex=True
                )
                
                # Magnitude and phase
                mag = torch.abs(spec).unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
                phase = torch.angle(spec)
                
                # Normalize magnitude
                mag_max = mag.max()
                mag_norm = mag / (mag_max + 1e-8)
                
                # Denoise
                mask = self.denoiser(mag_norm)
                
                # Apply mask
                mag_denoised = mag * mask.squeeze(0).squeeze(0)
                
                # Reconstruct
                spec_denoised = mag_denoised * torch.exp(1j * phase)
                spec_denoised = spec_denoised.squeeze(0).squeeze(0)
                
                # iSTFT
                audio_denoised = torch.istft(
                    spec_denoised,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window
                )
                
                denoised_channels.append(audio_denoised)
        
        denoised = torch.stack(denoised_channels)
        return denoised, C


# ============================================================================
# STAGE 2: AUDIO SOURCE LOCALIZATION (ASL)
# ============================================================================

class SELDNet(nn.Module):
    """
    Sound Event Localization and Detection Network
    Based on DCASE SELD architecture
    """
    def __init__(self, n_classes=8, n_channels=4):
        super().__init__()
        self.n_classes = n_classes
        
        # Shared CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Bidirectional GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Sound Event Detection (SED) head
        self.sed_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
            nn.Sigmoid()
        )
        
        # Direction of Arrival (DOA) head
        self.doa_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes * 3)  # [x, y, z] for each class
        )
    
    def forward(self, x):
        """
        Args:
            x: Multi-channel spectrogram [B, C, F, T]
        Returns:
            sed: Event detection [B, T, n_classes]
            doa: DOA estimates [B, T, n_classes, 3]
        """
        # CNN feature extraction
        B, C, F, T = x.shape
        features = self.cnn(x)  # [B, 256, F', T']
        
        # Collapse frequency dimension and permute for RNN: [B, T', 256]
        features = torch.mean(features, dim=2).permute(0, 2, 1)
        
        # Temporal modeling
        rnn_out, _ = self.gru(features)  # [B, T', 256]
        
        # Task-specific heads
        sed = self.sed_head(rnn_out)  # [B, T', n_classes]
        doa_raw = self.doa_head(rnn_out)  # [B, T', n_classes*3]
        doa = doa_raw.reshape(B, T_out, self.n_classes, 3)
        
        return sed, doa


class AudioSourceLocalizer:
    """Stage 2: Audio Source Localization (Multi-Channel Only)"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.classes = [
            'Gunshot', 'Explosion', 'Speech', 'Siren',
            'CarEngine', 'DogBark', 'BreakingGlass', 'Alarm'
        ]
        self.seld_net = SELDNet(n_classes=len(self.classes)).to(device)
        self.seld_net.eval()
        
        self.n_fft = 2048
        self.hop_length = 512
    
    def xyz_to_angles(self, xyz: torch.Tensor) -> Tuple[float, float]:
        """Convert Cartesian coordinates to spherical (azimuth, elevation)"""
        x, y, z = xyz[0].item(), xyz[1].item(), xyz[2].item()
        
        # Azimuth (horizontal angle)
        azimuth = np.degrees(np.arctan2(y, x))
        
        # Elevation (vertical angle)
        r = np.sqrt(x**2 + y**2 + z**2)
        elevation = np.degrees(np.arcsin(z / (r + 1e-8)))
        
        return azimuth, elevation
    
    def process(self, audio: torch.Tensor, sr: int) -> List[AudioEvent]:
        """
        Localize and detect sound events in multi-channel audio
        
        Args:
            audio: [C, T] multi-channel audio
            sr: Sample rate
            
        Returns:
            List of AudioEvent with spatial information
        """
        C = audio.shape[0]
        
        # Compute multi-channel spectrograms
        specs = []
        window = torch.hann_window(self.n_fft).to(self.device)
        
        for c in range(C):
            spec = torch.stft(
                audio[c].to(self.device),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True
            )
            specs.append(torch.abs(spec))
        
        # Stack channels: [C, F, T]
        multi_spec = torch.stack(specs).unsqueeze(0)  # [1, C, F, T]
        
        # Run SELD network
        with torch.no_grad():
            sed, doa = self.seld_net(multi_spec)  # [1, T', n_classes], [1, T', n_classes, 3]
        
        # Post-process detections
        events = []
        sed = sed[0].cpu()  # [T', n_classes]
        doa = doa[0].cpu()  # [T', n_classes, 3]
        
        threshold = 0.3
        for class_idx, class_name in enumerate(self.classes):
            class_probs = sed[:, class_idx]
            active_frames = (class_probs > threshold).numpy()
            
            if not active_frames.any():
                continue
            
            # Find continuous segments
            changes = np.diff(np.concatenate([[0], active_frames, [0]]).astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            for start, end in zip(starts, ends):
                # Compute average DOA for this segment
                avg_doa = doa[start:end, class_idx, :].mean(dim=0)
                azimuth, elevation = self.xyz_to_angles(avg_doa)
                
                # Time conversion
                time_scale = self.hop_length / sr
                start_time = start * time_scale
                end_time = end * time_scale
                confidence = class_probs[start:end].mean().item()
                
                events.append(AudioEvent(
                    label=class_name,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    azimuth=azimuth,
                    elevation=elevation
                ))
        
        return events


# ============================================================================
# STAGE 3: AUDIO EVENT CLASSIFICATION AND DETECTION
# ============================================================================

class CRNN_Classifier(nn.Module):
    """
    Convolutional Recurrent Neural Network for Sound Event Detection
    """
    def __init__(self, n_classes=8):
        super().__init__()
        
        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Spectrogram [B, 1, F, T]
        Returns:
            Frame-wise predictions [B, T, n_classes]
        """
        # CNN feature extraction
        B, _, F, T = x.shape
        features = self.conv(x)  # [B, 256, F', T']
        
        # Collapse frequency dimension and permute for RNN: [B, T', 256]
        features = torch.mean(features, dim=2).permute(0, 2, 1)
        
        # RNN
        rnn_out, _ = self.gru(features)
        
        # Classification
        logits = self.classifier(rnn_out)
        
        return logits


class AudioEventClassifier:
    """Stage 3: Audio Event Classification and Detection"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.classes = [
            'Gunshot', 'Explosion', 'Speech', 'Siren',
            'CarEngine', 'DogBark', 'BreakingGlass', 'Alarm'
        ]
        self.classifier = CRNN_Classifier(n_classes=len(self.classes)).to(device)
        self.classifier.eval()
        
        self.n_fft = 2048
        self.hop_length = 512
    
    def process(self, audio: torch.Tensor, sr: int) -> List[AudioEvent]:
        """
        Detect and classify audio events (works for mono or multi-channel)
        
        Args:
            audio: [C, T] audio tensor
            sr: Sample rate
            
        Returns:
            List of AudioEvent (without spatial info)
        """
        # Mix to mono if multi-channel
        if audio.shape[0] > 1:
            audio_mono = audio.mean(dim=0)
        else:
            audio_mono = audio[0]
        
        # Compute spectrogram
        window = torch.hann_window(self.n_fft).to(self.device)
        spec = torch.stft(
            audio_mono.to(self.device),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        
        mag = torch.abs(spec).unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
        
        # Normalize
        mag = (mag - mag.mean()) / (mag.std() + 1e-8)
        
        # Classify
        with torch.no_grad():
            logits = self.classifier(mag)  # [1, T', n_classes]
            probs = torch.sigmoid(logits[0]).cpu()  # [T', n_classes]
        
        # Post-process detections
        events = []
        threshold = 0.4
        
        for class_idx, class_name in enumerate(self.classes):
            class_probs = probs[:, class_idx]
            active_frames = (class_probs > threshold).numpy()
            
            if not active_frames.any():
                continue
            
            # Find continuous segments
            changes = np.diff(np.concatenate([[0], active_frames, [0]]).astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            for start, end in zip(starts, ends):
                time_scale = self.hop_length / sr
                start_time = start * time_scale
                end_time = end * time_scale
                confidence = class_probs[start:end].mean().item()
                
                events.append(AudioEvent(
                    label=class_name,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence
                ))
        
        return events


# ============================================================================
# STAGE 4: ADAPTIVE AUDIO SOURCE SEPARATION
# ============================================================================

class ConvTasNet(nn.Module):
    """
    Conv-TasNet for Blind Source Separation (Mono)
    """
    def __init__(self, N=512, L=16, B=128, H=512, P=3, X=8, R=3, n_sources=2):
        super().__init__()
        self.N = N
        self.L = L
        self.n_sources = n_sources
        
        # Encoder
        self.encoder = nn.Conv1d(1, N, kernel_size=L, stride=L//2, bias=False)
        
        # Separation module
        self.separator = nn.ModuleList([
            TemporalConvNet(N, B, H, P, X, R) for _ in range(n_sources)
        ])
        
        # Decoder
        self.decoder = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L//2, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: Waveform [B, 1, T]
        Returns:
            Separated sources [B, n_sources, T]
        """
        # Encode
        w = self.encoder(x)  # [B, N, T']
        
        # Separate
        masks = []
        for sep in self.separator:
            mask = sep(w)
            masks.append(mask)
        
        masks = torch.stack(masks, dim=1)  # [B, n_sources, N, T']
        
        # Apply masks
        w = w.unsqueeze(1)  # [B, 1, N, T']
        masked = w * masks  # [B, n_sources, N, T']
        
        # Decode
        sources = []
        for i in range(self.n_sources):
            s = self.decoder(masked[:, i])  # [B, 1, T]
            sources.append(s)
        
        return torch.stack(sources, dim=1).squeeze(2)  # [B, n_sources, T]


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network block"""
    def __init__(self, N, B, H, P, X, R):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalBlock(N, B, H, P, 2**x) for x in range(X) for _ in range(R)
        ])
        self.output = nn.Conv1d(N, N, kernel_size=1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        mask = self.activation(self.output(out))
        return mask


class TemporalBlock(nn.Module):
    """Temporal convolution block with dilation"""
    def __init__(self, N, B, H, P, dilation):
        super().__init__()
        self.conv1x1 = nn.Conv1d(N, B, kernel_size=1)
        self.prelu = nn.PReLU()
        self.norm = nn.GroupNorm(1, B)
        
        self.dconv = nn.Conv1d(
            B, B, kernel_size=P,
            padding=dilation * (P - 1) // 2,
            dilation=dilation,
            groups=B
        )
        self.output_conv = nn.Conv1d(B, N, kernel_size=1)
    
    def forward(self, x):
        residual = x
        x = self.conv1x1(x)
        x = self.prelu(x)
        x = self.norm(x)
        x = self.dconv(x)
        x = self.prelu(x)
        x = self.norm(x)
        x = self.output_conv(x)
        return x + residual


class BeamformingNetwork(nn.Module):
    """
    Spatially-informed separation using beamforming (Multi-channel)
    """
    def __init__(self, n_channels=4, n_sources=2):
        super().__init__()
        self.n_sources = n_sources
        
        # Spatial feature extractor
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Mask estimators for each source
        self.mask_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            ) for _ in range(n_sources)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Multi-channel spectrogram [B, C, F, T]
        Returns:
            Separated spectrograms [B, n_sources, F, T]
        """
        # Extract spatial features
        features = self.spatial_conv(x)
        
        # Estimate masks
        masks = []
        for estimator in self.mask_estimators:
            mask = estimator(features)
            masks.append(mask)
        
        masks = torch.stack(masks, dim=1).squeeze(2)  # [B, n_sources, F, T]
        
        # Apply to reference channel
        ref = torch.abs(x[:, 0:1, :, :])  # [B, 1, F, T]
        separated = masks * ref  # [B, n_sources, F, T]
        
        return separated


class AdaptiveSourceSeparator:
    """Stage 4: Adaptive Audio Source Separation"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Mono separation
        self.conv_tasnet = ConvTasNet(n_sources=3).to(device)
        self.conv_tasnet.eval()
        
        # Multi-channel separation
        self.beamformer = BeamformingNetwork(n_channels=4, n_sources=3).to(device)
        self.beamformer.eval()
        
        self.n_fft = 2048
        self.hop_length = 512
    
    def separate_mono(self, audio: torch.Tensor, events: List[AudioEvent]) -> List[torch.Tensor]:
        """Blind source separation for mono audio"""
        n_sources = len(events)
        
        if n_sources == 0:
            return []
        elif n_sources == 1:
            return [audio[0]]
        
        # Prepare input
        audio_input = audio[0].unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, T]
        
        # Separate
        with torch.no_grad():
            separated = self.conv_tasnet(audio_input)  # [1, n_sources, T]
        
        # Return individual sources
        sources = []
        for i in range(min(n_sources, separated.shape[1])):
            sources.append(separated[0, i].cpu())
        
        return sources
    
    def separate_multi(self, audio: torch.Tensor, events: List[AudioEvent]) -> List[torch.Tensor]:
        """Spatially-informed separation for multi-channel audio"""
        n_sources = len(events)
        
        if n_sources == 0:
            return []
        elif n_sources == 1:
            return [audio.mean(dim=0)]
        
        # Compute multi-channel spectrograms
        C = audio.shape[0]
        specs = []
        window = torch.hann_window(self.n_fft).to(self.device)
        
        for c in range(C):
            spec = torch.stft(
                audio[c].to(self.device),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True
            )
            specs.append(spec)
        
        # Stack: [1, C, F, T]
        multi_spec = torch.stack(specs).unsqueeze(0)
        
        # Get phase from reference channel
        phase = torch.angle(multi_spec[0, 0])
        
        # Separate (magnitude domain)
        with torch.no_grad():
            sep_mags = self.beamformer(torch.abs(multi_spec))  # [1, n_sources, F, T]
        
        # Reconstruct waveforms
        sources = []
        for i in range(min(n_sources, sep_mags.shape[1])):
            mag = sep_mags[0, i]
            spec_complex = mag * torch.exp(1j * phase)
            waveform = torch.istft(
                spec_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window
            )
            sources.append(waveform.cpu())
        
        return sources
    
    def process(self, audio: torch.Tensor, events: List[AudioEvent], is_multi_channel: bool) -> List[torch.Tensor]:
        """
        Adaptive separation based on channel count
        
        Args:
            audio: [C, T] audio tensor
            events: List of detected events
            is_multi_channel: True if C > 1
            
        Returns:
            List of separated waveforms
        """
        if is_multi_channel:
            return self.separate_multi(audio, events)
        else:
            return self.separate_mono(audio, events)


# ============================================================================
# STAGE 5: CLASS-SPECIFIC AUDIO ENHANCEMENT
# ============================================================================

class SpeechEnhancementGAN(nn.Module):
    """Speech enhancement using GAN-based approach"""
    def __init__(self):
        super().__init__()
        
        # Generator (U-Net)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=32, stride=2, padding=15),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=15),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=32, stride=2, padding=15),
                nn.PReLU()
            )
        ])
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=32, stride=2, padding=15),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(128, 32, kernel_size=32, stride=2, padding=15),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(64, 16, kernel_size=32, stride=2, padding=15),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(32, 1, kernel_size=32, stride=2, padding=15),
                nn.Tanh()
            )
        ])
    
    def forward(self, x):
        # Encoder with skip connections
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if i > 0:
                x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = layer(x)
        
        return x


class TransientRestorer(nn.Module):
    """Enhancement for transient sounds (gunshots, explosions)"""
    def __init__(self):
        super().__init__()
        
        self.enhancer = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=15, padding=7),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.enhancer(x)


class ClassSpecificEnhancer:
    """Stage 5: Class-Specific Audio Enhancement"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Enhancement models
        self.speech_enhancer = SpeechEnhancementGAN().to(device)
        self.speech_enhancer.eval()
        
        self.transient_restorer = TransientRestorer().to(device)
        self.transient_restorer.eval()
    
    def enhance_speech(self, waveform: torch.Tensor) -> torch.Tensor:
        """Enhance speech using GAN"""
        x = waveform.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            enhanced = self.speech_enhancer(x)
        
        return enhanced[0, 0].cpu()
    
    def enhance_transient(self, waveform: torch.Tensor) -> torch.Tensor:
        """Enhance transient sounds"""
        x = waveform.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            enhanced = self.transient_restorer(x)
        
        return enhanced[0, 0].cpu()
    
    def enhance_generic(self, waveform: torch.Tensor) -> torch.Tensor:
        """Generic enhancement (spectral smoothing)"""
        # Simple spectral enhancement
        return waveform * 1.2  # Gain boost
    
    def process(self, waveform: torch.Tensor, event_label: str) -> torch.Tensor:
        """
        Apply class-specific enhancement
        
        Args:
            waveform: [T] audio tensor
            event_label: Class label from Stage 3
            
        Returns:
            Enhanced waveform [T]
        """
        if event_label == 'Speech':
            return self.enhance_speech(waveform)
        elif event_label in ['Gunshot', 'Explosion', 'BreakingGlass']:
            return self.enhance_transient(waveform)
        else:
            return self.enhance_generic(waveform)


# ============================================================================
# INTEGRATED PIPELINE
# ============================================================================

class EnvironmentalAudioPipeline:
    """
    Complete End-to-End Pipeline:
    Input → Denoise → Localize → Classify → Separate → Enhance
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Initialize all stages
        print("[INFO] Initializing pipeline stages...")
        self.stage1 = AdaptivePreprocessor(device)
        self.stage2 = AudioSourceLocalizer(device)
        self.stage3 = AudioEventClassifier(device)
        self.stage4 = AdaptiveSourceSeparator(device)
        self.stage5 = ClassSpecificEnhancer(device)
        print("[INFO] Pipeline ready!\n")
    
    def process(self, audio_path: str, output_dir: str = './output') -> Dict:
        """
        Process audio file through complete pipeline
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory for output files
            
        Returns:
            Dictionary with results
        """
        print(f"[INFO] Processing: {audio_path}")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load audio
        # audio, sr = torchaudio.load(audio_path)
        data, sr = sf.read(audio_path)
        # Ensure it's (C, T)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        audio = torch.from_numpy(data).float()
        print(f"   Loaded: {audio.shape[0]} channels, {sr} Hz, {audio.shape[1]/sr:.2f}s duration")
        
        # ====== STAGE 1: Adaptive Pre-processing ======
        print("\n[STAGE 1] Denoising...")
        denoised, C = self.stage1.process(audio, sr)
        print(f"   [OK] Denoised {C} channel(s)")
        
        # ====== STAGE 2: Source Localization (if multi-channel) ======
        spatial_events = []
        if C > 1:
            print("\n[STAGE 2] Source Localization...")
            spatial_events = self.stage2.process(denoised, sr)
            print(f"   [OK] Localized {len(spatial_events)} sources")
            for evt in spatial_events:
                print(f"     - {evt.label}: Az={evt.azimuth:.1f}deg, El={evt.elevation:.1f}deg")
        else:
            print("\n[STAGE 2] Skipped (mono input)")
        
        # ====== STAGE 3: Event Classification ======
        print("\n[STAGE 3] Event Classification...")
        classified_events = self.stage3.process(denoised, sr)
        
        # Merge spatial info if available
        if spatial_events:
            for i, evt in enumerate(classified_events):
                if i < len(spatial_events):
                    evt.azimuth = spatial_events[i].azimuth
                    evt.elevation = spatial_events[i].elevation
        
        print(f"   [OK] Detected {len(classified_events)} events:")
        for evt in classified_events:
            spatial_info = f" @ ({evt.azimuth:.0f}deg, {evt.elevation:.0f}deg)" if evt.azimuth else ""
            print(f"     - {evt.label}: {evt.start_time:.2f}s - {evt.end_time:.2f}s{spatial_info}")
        
        # ====== STAGE 4: Source Separation ======
        print("\n[STAGE 4] Source Separation...")
        separated_waveforms = self.stage4.process(denoised, classified_events, C > 1)
        print(f"   [OK] Separated {len(separated_waveforms)} sources")
        
        # ====== STAGE 5: Enhancement ======
        print("\n[STAGE 5] Class-Specific Enhancement...")
        separated_sources = []
        
        for i, (waveform, event) in enumerate(zip(separated_waveforms, classified_events)):
            enhanced = self.stage5.process(waveform, event.label)
            
            # Save enhanced audio
            filename = f"source_{i+1}_{event.label}_{event.start_time:.1f}s.wav"
            filepath = output_path / filename
            # torchaudio.save(str(filepath), enhanced.unsqueeze(0), sr)
            sf.write(str(filepath), enhanced.detach().cpu().numpy(), sr)
            
            separated_sources.append(SeparatedSource(
                waveform=enhanced,
                sample_rate=sr,
                event=event,
                enhanced=True
            ))
            
            print(f"   [OK] Enhanced {event.label} -> {filename}")
        
        # Compile results
        results = {
            'input_file': audio_path,
            'channels': C,
            'sample_rate': sr,
            'events': classified_events,
            'separated_sources': separated_sources,
            'output_directory': str(output_path)
        }
        
        print(f"\n[INFO] Processing complete! Output saved to: {output_path}")
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of the pipeline"""
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    pipeline = EnvironmentalAudioPipeline(device=device)
    
    # Create synthetic test audio (3 seconds, mono)
    print("Creating synthetic test audio...")
    sr = 16000
    duration = 3.0
    t = torch.linspace(0, duration, int(sr * duration))
    
    # Simulate: speech (1-2s) + siren (1.5-2.5s)
    speech = 0.3 * torch.sin(2 * np.pi * 200 * t) * (t > 1) * (t < 2)
    siren = 0.4 * torch.sin(2 * np.pi * 800 * t + 50 * torch.sin(2 * np.pi * 3 * t))
    siren = siren * (t > 1.5) * (t < 2.5)
    
    audio = (speech + siren).unsqueeze(0)  # [1, T] mono
    
    # Save test file
    # Save test file
    test_path = './test_audio.wav'
    # torchaudio.save(test_path, audio, sr)
    sf.write(test_path, audio.squeeze(0).numpy(), sr)
    
    # Process
    results = pipeline.process(test_path, output_dir='./outputs/c')
    
    # Display summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Input: {results['input_file']}")
    print(f"Channels: {results['channels']}")
    print(f"Sample Rate: {results['sample_rate']} Hz")
    print(f"\nDetected Events: {len(results['events'])}")
    for evt in results['events']:
        print(f"  - {evt.label}: {evt.start_time:.2f}s - {evt.end_time:.2f}s")
    print(f"\nSeparated & Enhanced Sources: {len(results['separated_sources'])}")
    print(f"Output Directory: {results['output_directory']}")
    print("="*60)


if __name__ == '__main__':
    main()

    