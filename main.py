# main.py
import os
import sys

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Unbuffered output + quieter logs for pyannote/speechbrain/tokenizers
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYANNOTE_AUDIO_LOG_LEVEL"] = "ERROR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import argparse
import torch.nn.functional as F
from datetime import datetime, timezone

from src.model import CNNMultistreamModel
from src.audio_detector import detect_audio_deepfake
from src.utils import (
    get_device, process_single_video, build_single_video_tensors_fromdisk,
    default_phoneme_vocab, IMAGES_PER_PHON
)

# ==== weights/model hyperparams ====
WEIGHTS_PATH = "checkpoints/best_model.pth"
EMBED_DIM    = 128
NUM_HEADS    = 4
FPS          = 25.0
LABEL_FOLDER = "testvideo"  # subfolder under --outdir
# ===============================================

def load_weights(model: torch.nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt

def main():
    ap = argparse.ArgumentParser(description="Single-video deepfake test (audio-visual)")
    ap.add_argument("--video", required=True, help="Path to input video (.mp4/.mov)")
    ap.add_argument("--outdir", required=True, help="Where to save crops/json/results")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    device = get_device()
    print(f"[INFO] Using device: {device}", flush=True)

    aligned_json, label_dir, raw_audio, has_audio = process_single_video(
        video_path=args.video,
        output_dir=outdir,
        label=LABEL_FOLDER,
        fps=FPS,
        whisper_model_name="large-v2",
        language="en",
        compute_type="float32"
    )
    print(f"[OK] Aligned JSON: {aligned_json}", flush=True)

    # 1) Visual / Viseme Model
    phoneme_vocab = default_phoneme_vocab()
    geoms, imgs, arcs, mask = build_single_video_tensors_fromdisk(
        aligned_json_path=aligned_json,
        label_dir=label_dir,
        phoneme_vocab=phoneme_vocab,
        device=torch.device(device)
    )

    model = CNNMultistreamModel(
        arcface_dim=512, geo_dim=1,
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        frames_per_phon=IMAGES_PER_PHON
    ).to(device)
    load_weights(model, WEIGHTS_PATH, device)
    model.eval()

    with torch.no_grad():
        logits, _ = model(geoms, imgs, arcs, mask)
        v_probs = F.softmax(logits, dim=1).cpu().numpy().tolist()[0]
        v_pred_label = int(v_probs[1] > 0.5)  # 1=Fake, 0=Real

    num_matched_frames = int(mask.sum().item())
    with open(aligned_json, "r", encoding="utf-8") as f:
        aligned_frames = json.load(f)
    has_phonemes = any(str(f.get("phoneme", "")).lower() not in {"<sil>", "sp", "", " ", "none"} for f in aligned_frames)

    # 2) Audio Deepfake Detection
    if has_audio and raw_audio is not None:
        print("[STEP] Running Audio Deepfake Detection (Wav2Vec2)...", flush=True)
        audio_res = detect_audio_deepfake(raw_audio, sampling_rate=16000, device=device)
        print(f"[OK] Audio verdict: {audio_res['verdict']} (P(fake)={audio_res['p_fake']:.4f})", flush=True)
    else:
        print("[INFO] No audio speech track in video. Skipping audio deepfake classification.", flush=True)
        audio_res = {
            "has_audio": False,
            "has_speech": False,
            "p_fake": 0.0,
            "p_real": 0.0,
            "fake": False,
            "verdict": "No audio track present",
            "observation": "Video contains no audio stream (visual forensic analysis only)."
        }

    # 3) Multi-modal Decision Fusion
    v_fake = float(v_probs[1])
    v_real = float(v_probs[0])

    if audio_res.get("has_speech", False):
        a_fake = float(audio_res["p_fake"])
        # If either visual or audio indicates strong deepfake manipulation (>0.65), flag overall media
        if v_fake > 0.65 or a_fake > 0.65:
            final_p_fake = max(v_fake, a_fake)
        else:
            final_p_fake = 0.5 * v_fake + 0.5 * a_fake
        final_p_real = 1.0 - final_p_fake
        final_fake = final_p_fake > 0.5
        mode = "combined"
    else:
        # Video without audio or non-speech video: decisions driven by visual forensic analysis
        final_p_fake = v_fake
        final_p_real = v_real
        final_fake = v_fake > 0.5
        mode = "visual"

    final_pred_label = 1 if final_fake else 0
    label_map = {0: "Real", 1: "Fake"}

    result = {
        "model": "PIA + Wav2Vec2",
        "code_reference": "https://github.com/skrantidatta/PIA",
        "paper_reference": "https://arxiv.org/pdf/2510.14241",
        "video": os.path.abspath(args.video),
        "mode": mode,
        "has_audio": has_audio,
        "has_speech": audio_res.get("has_speech", False),
        "has_phonemes": has_phonemes,
        "matched_phoneme_frames": num_matched_frames,
        "predicted_label": final_pred_label,
        "predicted_label_name": label_map[final_pred_label],
        "fake": final_fake,
        "probs": {"real": round(final_p_real, 4), "fake": round(final_p_fake, 4)},
        "video_analysis": {
            "p_fake": round(v_fake, 4),
            "p_real": round(v_real, 4),
            "fake": v_fake > 0.5,
            "mode": "phoneme_viseme" if has_phonemes else "visual_temporal",
            "description": "Phoneme-viseme lip sync consistency" if has_phonemes else "Facial landmarks, temporal dynamics and ArcFace representations"
        },
        "audio_analysis": audio_res,
        "weights": WEIGHTS_PATH,
        "aligned_json": aligned_json,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    print(f"[RESULT] Overall verdict: {result['predicted_label_name']} (P(fake)={final_p_fake:.4f}, Mode={mode})", flush=True)
    if audio_res.get("has_speech", False):
        print(f"[RESULT] Visual Fake: {v_fake*100:.1f}%, Audio Fake: {audio_res['p_fake']*100:.1f}%", flush=True)
    else:
        print(f"[RESULT] Visual Fake: {v_fake*100:.1f}% (No audio track)", flush=True)

    results_path = os.path.join(outdir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] {results_path}", flush=True)

if __name__ == "__main__":
    main()
