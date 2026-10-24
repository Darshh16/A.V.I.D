# src/audio_detector.py
"""
Audio deepfake detection module using fine-tuned Wav2Vec2.
Detects AI-generated, synthetic, and voice-cloned speech.
"""
import os
import torch
import numpy as np
from typing import Dict, Any, Optional

_AUDIO_MODEL = None
_FEATURE_EXTRACTOR = None
MODEL_ID = "mo-thecreator/Deepfake-audio-detection"


def get_audio_detector(device: Optional[str] = None):
    global _AUDIO_MODEL, _FEATURE_EXTRACTOR
    if _AUDIO_MODEL is None or _FEATURE_EXTRACTOR is None:
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading audio deepfake detector ({MODEL_ID}) on {device}...", flush=True)
        _FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        _AUDIO_MODEL = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
        _AUDIO_MODEL.to(device)
        _AUDIO_MODEL.eval()
    return _AUDIO_MODEL, _FEATURE_EXTRACTOR


def detect_audio_deepfake(
    audio: np.ndarray,
    sampling_rate: int = 16000,
    device: Optional[str] = None,
    chunk_duration_sec: float = 6.0
) -> Dict[str, Any]:
    """
    Run deepfake detection on an audio waveform.
    audio: 1D numpy array of float32 samples.
    Returns dictionary with fake/real probabilities and verdict.
    """
    if audio is None or len(audio) == 0:
        return {
            "has_audio": False,
            "has_speech": False,
            "p_fake": 0.0,
            "p_real": 0.0,
            "fake": False,
            "verdict": "No audio present",
            "observation": "Video contains no audio stream."
        }

    # Check for near-silence or zero signal
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 1e-4 or len(audio) < sampling_rate * 0.5:
        return {
            "has_audio": True,
            "has_speech": False,
            "p_fake": 0.0,
            "p_real": 1.0,
            "fake": False,
            "verdict": "Silent audio (no speech)",
            "observation": "Audio stream is silent or contains no audible speech."
        }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, fe = get_audio_detector(device)

    # Chunk audio into segments of chunk_duration_sec to evaluate over time
    chunk_size = int(sampling_rate * chunk_duration_sec)
    hop_size = int(chunk_size * 0.75)

    if len(audio) <= chunk_size:
        chunks = [audio]
    else:
        chunks = []
        for start in range(0, len(audio) - chunk_size + 1, hop_size):
            chunks.append(audio[start:start + chunk_size])
        if len(audio) % hop_size != 0 and (len(audio) - len(chunks) * hop_size) > sampling_rate:
            chunks.append(audio[-chunk_size:])

    chunk_fake_probs = []
    with torch.no_grad():
        for ch in chunks:
            inputs = fe(ch, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            # mo-thecreator/Deepfake-audio-detection: id2label {0: 'fake', 1: 'real'}
            p_fake = float(probs[0])
            chunk_fake_probs.append(p_fake)

    if not chunk_fake_probs:
        p_fake_final = 0.0
    else:
        # Use a combination of average and peak detection (if a segment is clearly fake)
        avg_fake = float(np.mean(chunk_fake_probs))
        max_fake = float(np.max(chunk_fake_probs))
        # Weighted ensemble: 60% average, 40% max to catch localized deepfake voice cloning
        p_fake_final = 0.6 * avg_fake + 0.4 * max_fake

    p_real_final = float(max(0.0, min(1.0, 1.0 - p_fake_final)))
    fake = p_fake_final > 0.5

    if fake:
        obs = f"Synthetic/AI voice artifacts detected (acoustic confidence: {p_fake_final*100:.1f}% fake)."
        verdict = "Synthetic / AI-Generated Voice"
    else:
        obs = f"Acoustic features consistent with organic human speech ({p_real_final*100:.1f}% real)."
        verdict = "Authentic Human Voice"

    return {
        "has_audio": True,
        "has_speech": True,
        "p_fake": round(p_fake_final, 4),
        "p_real": round(p_real_final, 4),
        "fake": fake,
        "verdict": verdict,
        "observation": obs,
        "rms": round(rms, 5),
        "num_chunks_analyzed": len(chunks)
    }
