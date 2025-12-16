import unittest
import torch
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Adjust path to import from sibling directory 'c'
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from c.cl1 import (
    AdaptivePreprocessor,
    AudioSourceLocalizer,
    AudioEventClassifier,
    AdaptiveSourceSeparator,
    ClassSpecificEnhancer,
    EnvironmentalAudioPipeline,
    AudioEvent
)

class TestEcoSenseAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use CPU for testing to ensure compatibility and speed
        cls.device = 'cpu'
        cls.sr = 16000
    
    def setUp(self):
        # Suppress prints during tests to keep output clean
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        
    def tearDown(self):
        sys.stdout.close()
        sys.stdout = self._original_stdout

    def test_01_preprocessor_mono(self):
        """Test Stage 1 with Mono Input"""
        # Create dummy mono audio [1, 16000] (1 second)
        audio = torch.randn(1, 16000)
        preprocessor = AdaptivePreprocessor(device=self.device)
        
        denoised, channels = preprocessor.process(audio, self.sr)
        
        self.assertEqual(channels, 1)
        self.assertEqual(denoised.shape, audio.shape)
        # Ensure no NaNs introduced
        self.assertFalse(torch.isnan(denoised).any())

    def test_02_preprocessor_multi(self):
        """Test Stage 1 with Multi-channel Input"""
        # Create dummy multi-channel audio [4, 16000]
        audio = torch.randn(4, 16000)
        preprocessor = AdaptivePreprocessor(device=self.device)
        
        denoised, channels = preprocessor.process(audio, self.sr)
        
        self.assertEqual(channels, 4)
        self.assertEqual(denoised.shape, audio.shape)

    def test_03_localizer(self):
        """Test Stage 2: Source Localization"""
        # Localizer expects multi-channel input
        audio = torch.randn(4, 32000) # 2 seconds
        localizer = AudioSourceLocalizer(device=self.device)
        
        events = localizer.process(audio, self.sr)
        
        # Even if no events are detected (random noise), it should return a list
        self.assertIsInstance(events, list)
        
        # Verify network forward pass dimensions
        specs = torch.randn(1, 4, 128, 100).to(self.device) # [B, C, F, T]
        sed, doa = localizer.seld_net(specs)
        
        self.assertEqual(sed.shape[0], 1) # Batch
        self.assertEqual(sed.shape[2], 8) # Classes
        self.assertEqual(doa.shape[3], 3) # XYZ coordinates

    def test_04_classifier(self):
        """Test Stage 3: Event Classification"""
        audio = torch.randn(1, 32000)
        classifier = AudioEventClassifier(device=self.device)
        
        events = classifier.process(audio, self.sr)
        self.assertIsInstance(events, list)
        
        # Test network forward pass
        spec = torch.randn(1, 1, 128, 100).to(self.device)
        logits = classifier.classifier(spec)
        self.assertEqual(logits.shape[2], 8) # 8 classes

    def test_05_separator_mono(self):
        """Test Stage 4: Separation (Mono)"""
        audio = torch.randn(1, 16000)
        separator = AdaptiveSourceSeparator(device=self.device)
        
        # Mock detected events
        events = [
            AudioEvent(label='Speech', start_time=0.0, end_time=1.0, confidence=0.9),
            AudioEvent(label='Siren', start_time=0.0, end_time=1.0, confidence=0.8)
        ]
        
        sources = separator.process(audio, events, is_multi_channel=False)
        
        self.assertEqual(len(sources), 2)
        # ConvTasNet output length matches input length
        self.assertEqual(sources[0].shape[-1], audio.shape[-1])

    def test_06_separator_multi(self):
        """Test Stage 4: Separation (Multi-channel)"""
        audio = torch.randn(4, 16000)
        separator = AdaptiveSourceSeparator(device=self.device)
        
        events = [AudioEvent(label='Speech', start_time=0.0, end_time=1.0, confidence=0.9)]
        
        sources = separator.process(audio, events, is_multi_channel=True)
        
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].dim(), 1)

    def test_07_enhancer(self):
        """Test Stage 5: Enhancement"""
        waveform = torch.randn(16000)
        enhancer = ClassSpecificEnhancer(device=self.device)
        
        # Test Speech GAN
        out_speech = enhancer.process(waveform, 'Speech')
        self.assertEqual(out_speech.shape, waveform.shape)
        
        # Test Transient Model
        out_transient = enhancer.process(waveform, 'Gunshot')
        self.assertEqual(out_transient.shape, waveform.shape)

    @patch('torchaudio.load')
    @patch('torchaudio.save')
    def test_08_full_pipeline(self, mock_save, mock_load):
        """Test Full Pipeline Integration"""
        # Mock audio load return value: (waveform, sample_rate)
        mock_audio = torch.randn(1, 32000)
        mock_load.return_value = (mock_audio, 16000)
        
        pipeline = EnvironmentalAudioPipeline(device=self.device)
        
        # Run pipeline
        results = pipeline.process("dummy_input.wav", output_dir="test_output")
        
        # Verify structure of results
        self.assertEqual(results['channels'], 1)
        self.assertEqual(results['sample_rate'], 16000)
        self.assertIsInstance(results['events'], list)
        self.assertIsInstance(results['separated_sources'], list)

if __name__ == '__main__':
    unittest.main()