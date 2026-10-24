#!/usr/bin/env python3
"""
A.V.I.D. local backend.

Put this file and index.html inside your cloned PIA folder (next to main.py), then run:

    pip install flask
    python server.py

and open http://localhost:8000

It receives an uploaded video, runs PIA (python main.py --video ... --outdir ...),
and returns the verdict as JSON for the UI.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.py"
WEIGHTS = ROOT / "checkpoints" / "best_model.pth"
ALLOWED = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MAX_MB = 200
TIMEOUT_S = 60 * 60          # PIA can be slow on CPU
CLOSURE_PHONEMES = {"m", "b", "p"}   # same set PIA uses for lip-closure checks
PORT = 8000

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
job_lock = threading.Lock()


@app.after_request
def add_cors(resp):
    # lets the UI work even if index.html is opened straight from disk
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def to_mp4(src: Path, dst: Path) -> Path:
    """PIA expects mp4. Convert other containers with ffmpeg."""
    if src.suffix.lower() == ".mp4":
        return src
    if not shutil.which("ffmpeg"):
        raise RuntimeError("This file is not an MP4 and ffmpeg is not installed to convert it. "
                           "Upload an MP4 or install ffmpeg.")
    cmd = ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-c:a", "aac", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg could not convert the video:\n" + proc.stderr[-800:])
    return dst


def build_evidence(aligned_path: Path, res: dict):
    """Turn PIA's per-frame alignment file and multi-modal analysis into timeline evidence."""
    frames = json.loads(aligned_path.read_text(encoding="utf-8"))
    total = len(frames)
    closure_frames = [f for f in frames if str(f.get("phoneme", "")).lower() in CLOSURE_PHONEMES]
    flagged = [f for f in frames if f.get("mismatch")]

    groups = []
    for f in flagged:
        if groups and f["timestamp"] - groups[-1][-1]["timestamp"] <= 0.25:
            groups[-1].append(f)
        else:
            groups.append([f])
    groups = sorted(groups, key=len, reverse=True)[:8]
    groups.sort(key=lambda g: g[0]["timestamp"])

    p_fake = float(res["probs"]["fake"])
    fake = bool(res.get("fake", p_fake > 0.5))
    v_analysis = res.get("video_analysis", {})
    a_analysis = res.get("audio_analysis", {})
    v_fake = float(v_analysis.get("p_fake", p_fake))
    has_audio = res.get("has_audio", False)
    has_speech = res.get("has_speech", False)
    has_phonemes = res.get("has_phonemes", False)

    evidence = []

    # 1. Visual Forensic Signal
    if has_phonemes:
        evidence.append({
            "time": "00:00",
            "signal": "Visual (Lip-sync)",
            "observation": (f"Lip-sync classifier: {v_fake * 100:.1f}% fake, {(1 - v_fake) * 100:.1f}% real. "
                            f"{len(flagged)} of {len(closure_frames)} lip-closure frames (sounds P/B/M) showed open lips."),
            "level": "Review" if v_fake > 0.5 else "Info",
        })
    else:
        evidence.append({
            "time": "00:00",
            "signal": "Visual Forensics",
            "observation": (f"Visual facial model: {v_fake * 100:.1f}% fake, {(1 - v_fake) * 100:.1f}% real. "
                            f"Analyzed facial landmarks, temporal dynamics and ArcFace embeddings across timeline."),
            "level": "Review" if v_fake > 0.5 else "Info",
        })

    # 2. Audio Forensic Signal
    if has_audio and has_speech:
        a_fake = float(a_analysis.get("p_fake", 0.0))
        evidence.append({
            "time": "00:00",
            "signal": "Audio Forensics",
            "observation": f"Acoustic Deepfake Detector: {a_fake * 100:.1f}% fake. {a_analysis.get('observation', '')}",
            "level": "Review" if a_fake > 0.5 else "Info",
        })
    else:
        evidence.append({
            "time": "00:00",
            "signal": "Audio Forensics",
            "observation": "Video does not contain an active speech audio stream. Analysis performed on visual signals.",
            "level": "Info",
        })

    # 3. Flagged Lip-sync moments
    for g in groups:
        ph = g[0].get("phoneme", "?")
        lowest = min(x.get("closure_score", 0.0) for x in g)
        level = "Info"
        if fake:
            level = "Review" if len(g) >= 3 else "Moderate"
        evidence.append({
            "time": fmt_time(g[0]["timestamp"]),
            "signal": "Lip-sync Mismatch",
            "observation": f"Lips stayed open (closure score {lowest:.2f}) while speaker said '{ph}' ({len(g)} frames).",
            "level": level,
        })

    return evidence, total, len(closure_frames), len(flagged)


@app.route("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.route("/health")
def health():
    return jsonify(ok=True, pia_found=MAIN.exists(), weights_found=WEIGHTS.exists(), busy=job_lock.locked())


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return ("", 204)
    if not MAIN.exists():
        return jsonify(error="main.py not found. Put server.py inside your cloned PIA folder."), 500
    if not WEIGHTS.exists():
        return jsonify(error="checkpoints/best_model.pth not found in the PIA folder."), 500

    up = request.files.get("video")
    if up is None or not up.filename:
        return jsonify(error="No video was uploaded."), 400
    suffix = Path(up.filename).suffix.lower()
    if suffix not in ALLOWED:
        return jsonify(error=f"Unsupported file type '{suffix}'. Use MP4, MOV, MKV, AVI or WEBM."), 400

    if not job_lock.acquire(blocking=False):
        return jsonify(error="Another analysis is still running. Wait for it to finish and try again."), 429

    work = Path(tempfile.mkdtemp(prefix="avid_"))
    try:
        src = work / f"input{suffix}"
        up.save(src)
        video = to_mp4(src, work / "input.mp4")
        outdir = work / "out"

        proc = subprocess.run(
            [sys.executable, str(MAIN), "--video", str(video), "--outdir", str(outdir)],
            cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        results_file = outdir / "results.json"
        if proc.returncode != 0 or not results_file.exists():
            tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:])
            return jsonify(
                error=("Could not complete analysis on this video. Make sure a face is visible in the video frames "
                       "and required packages are installed."),
                details=tail,
            ), 500

        res = json.loads(results_file.read_text(encoding="utf-8"))
        aligned = Path(res["aligned_json"])

        p_fake = float(res["probs"]["fake"])
        fake = bool(res.get("fake", p_fake > 0.5))
        evidence, total, closure, flagged = build_evidence(aligned, res)

        v_score = round(float(res.get("video_analysis", {}).get("p_fake", p_fake)) * 100)
        has_audio = res.get("has_audio", False)
        a_score = round(float(res.get("audio_analysis", {}).get("p_fake", 0.0)) * 100) if has_audio else None

        return jsonify(
            fake=fake,
            verdict="Deepfake detected" if fake else "Real (no manipulation found)",
            score=round(p_fake * 100),
            probs=res["probs"],
            video_score=v_score,
            audio_score=a_score,
            has_audio=has_audio,
            has_speech=res.get("has_speech", False),
            mode=res.get("mode", "combined"),
            evidence=evidence,
            frames_total=total,
            closure_frames=closure,
            mismatch_frames=flagged,
            duration=total / 25.0,
            model=res.get("model", "PIA + Wav2Vec2"),
            video_analysis=res.get("video_analysis"),
            audio_analysis=res.get("audio_analysis"),
        )
    except subprocess.TimeoutExpired:
        return jsonify(error="Analysis timed out. Try a shorter clip."), 504
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 500
    finally:
        job_lock.release()
        shutil.rmtree(work, ignore_errors=True)


@app.errorhandler(413)
def too_big(_):
    return jsonify(error=f"File is larger than {MAX_MB} MB."), 413


if __name__ == "__main__":
    print(f"A.V.I.D. backend running on http://localhost:{PORT}")
    app.run(host="127.0.0.1", port=PORT, threaded=True)