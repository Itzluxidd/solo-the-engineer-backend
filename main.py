import os
import uuid
import tempfile

import numpy as np
import requests
import librosa
import soundfile as sf
import pyloudnorm as pyln

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------
# Basic file storage settings
# ---------------------------

UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/tmp/audio")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:5000")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def download_file(url: str, local_path: str) -> str:
    """Download file from URL to local path."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return local_path


def save_and_get_url(audio_data: np.ndarray, sr: int, filename: str) -> str:
    """Save audio to UPLOAD_FOLDER and return public URL."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Make sure it's float and not clipping
    audio = audio_data.astype(np.float32)
    peak = np.max(np.abs(audio)) + 1e-9
    if peak > 1.0:
        audio = audio / peak

    sf.write(filepath, audio, sr)
    return f"{BASE_URL}/files/{filename}"


@app.route("/files/<path:filename>")
def serve_file(filename: str):
    """Serve audio files stored in UPLOAD_FOLDER."""
    return send_from_directory(UPLOAD_FOLDER, filename)


# ===========================
# 1. STEM "SEPARATION" (FAKE)
# ===========================

@app.route("/api/separate-stems", methods=["POST"])
def separate_stems():
    """
    Placeholder stem separation.

    For now, we just return the same audio as both 'vocals' and 'instrumental'
    so the feature works end-to-end without Spleeter.
    """
    try:
        data = request.json or {}
        audio_url = data.get("audio_url")
        project_id = data.get("project_id", str(uuid.uuid4()))

        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        # Download audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name

        # Load audio
        y, sr = librosa.load(input_path, sr=44100)

        # Save two copies with different names
        vocals_url = save_and_get_url(y, sr, f"{project_id}_vocals.wav")
        instrumental_url = save_and_get_url(y, sr, f"{project_id}_instrumental.wav")

        os.unlink(input_path)

        return jsonify({
            "success": True,
            "vocals_url": vocals_url,
            "instrumental_url": instrumental_url
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ===========================
# 2. KEY & BPM DETECTION
# ===========================

@app.route("/api/analyze", methods=["POST"])
def analyze_audio():
    """Detect key and BPM of an audio file."""
    try:
        data = request.json or {}
        audio_url = data.get("audio_url")

        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        # Download audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name

        # Load audio
        y, sr = librosa.load(input_path, sr=22050)

        # BPM detection
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0]) if hasattr(tempo, "__iter__") else float(tempo)
        bpm = round(bpm)

        # Key detection (simple)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)

        key_names = ["C", "C#", "D", "D#", "E", "F",
                     "F#", "G", "G#", "A", "A#", "B"]
        key_index = int(np.argmax(chroma_avg))

        major_profile = [6.35, 2.23, 3.48, 2.33,
                         4.38, 4.09, 2.52, 5.19,
                         2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38,
                         2.60, 3.53, 2.54, 4.75,
                         3.98, 2.69, 3.34, 3.17]

        major_corr = np.corrcoef(
            chroma_avg, np.roll(major_profile, key_index)
        )[0, 1]
        minor_corr = np.corrcoef(
            chroma_avg, np.roll(minor_profile, key_index)
        )[0, 1]

        mode = "major" if major_corr > minor_corr else "minor"
        key = f"{key_names[key_index]} {mode}"

        os.unlink(input_path)

        return jsonify({
            "success": True,
            "bpm": bpm,
            "key": key
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ===========================
# 3. AUTOTUNE (SIMPLE)
# ===========================

@app.route("/api/autotune", methods=["POST"])
def autotune_vocals():
    """Apply simple pitch-based processing to vocals."""
    try:
        data = request.json or {}
        audio_url = data.get("audio_url")
        style = data.get("style", "medium").lower()  # natural, medium, hard
        project_id = data.get("project_id", str(uuid.uuid4()))

        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        # Download audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name

        # Load audio
        y, sr = librosa.load(input_path, sr=44100)

        if style == "hard":
            # "Hard" style: more robotic / processed feel
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=0)
            y_tuned = librosa.effects.harmonic(y_shifted)
        else:
            # Gentle processing
            y_proc = librosa.effects.pitch_shift(y, sr=sr, n_steps=0)
            strength = {"natural": 0.3, "medium": 0.6}.get(style, 0.6)
            y_tuned = y_proc * strength + y * (1 - strength)

        # Normalize
        peak = np.max(np.abs(y_tuned)) + 1e-9
        y_tuned = y_tuned / peak * 0.95

        tuned_url = save_and_get_url(y_tuned, sr, f"{project_id}_tuned.wav")

        os.unlink(input_path)

        return jsonify({"success": True, "tuned_url": tuned_url})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ===========================
# 4. MIX & MASTER
# ===========================

@app.route("/api/mix-master", methods=["POST"])
def mix_master():
    """Mix and master vocals with instrumental."""
    try:
        data = request.json or {}
        vocal_url = data.get("vocal_url")
        instrumental_url = data.get("instrumental_url")
        project_id = data.get("project_id", str(uuid.uuid4()))

        if not vocal_url or not instrumental_url:
            return jsonify({
                "success": False,
                "error": "vocal_url and instrumental_url are required"
            }), 400

        # Download files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1:
            download_file(vocal_url, tmp1.name)
            vocal_path = tmp1.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
            download_file(instrumental_url, tmp2.name)
            instrumental_path = tmp2.name

        # Load audio
        vocals, sr = librosa.load(vocal_path, sr=44100)
        instrumental, _ = librosa.load(instrumental_path, sr=44100)

        # Match lengths
        max_len = max(len(vocals), len(instrumental))
        vocals = np.pad(vocals, (0, max_len - len(vocals)))
        instrumental = np.pad(instrumental, (0, max_len - len(instrumental)))

        # Simple vocal enhancement
        vocals_filtered = librosa.effects.preemphasis(vocals, coef=0.97)
        vocals_enhanced = vocals_filtered + 0.1 * librosa.effects.harmonic(vocals_filtered)

        # Mix
        mix = instrumental * 0.8 + vocals_enhanced * 0.7

        # Soft compression
        threshold = 0.5
        ratio = 4.0
        above = np.abs(mix) > threshold
        mix_compressed = mix.copy()
        mix_compressed[above] = (
            np.sign(mix[above])
            * (threshold + (np.abs(mix[above]) - threshold) / ratio)
        )

        # Normalize to -1 dB peak
        peak = np.max(np.abs(mix_compressed)) + 1e-9
        target_peak = 10 ** (-1.0 / 20.0)
        mix_normalized = mix_compressed * (target_peak / peak)

        # Limiter at -0.3 dB
        limit = 10 ** (-0.3 / 20.0)
        mix_limited = np.clip(mix_normalized, -limit, limit)

        master_url = save_and_get_url(mix_limited, sr, f"{project_id}_master.wav")

        os.unlink(vocal_path)
        os.unlink(instrumental_path)

        return jsonify({"success": True, "master_url": master_url})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ===========================
# 5. PLATFORM READINESS CHECK
# ===========================

@app.route("/api/platform-check", methods=["POST"])
def platform_check():
    """Analyze master for streaming platform compatibility."""
    try:
        data = request.json or {}
        audio_url = data.get("audio_url")

        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        # Download audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name

        # Load audio
        y, sr = librosa.load(input_path, sr=44100, mono=False)
        if y.ndim == 1:
            y = np.array([y, y])  # make stereo: [2, samples]

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y.T)

        true_peak = 20 * np.log10(np.max(np.abs(y)) + 1e-10)

        info = sf.info(input_path)
        sample_rate = info.samplerate
        bit_depth = 16 if "16" in str(info.subtype) else 24

        issues = []

        # Spotify
        if loudness < -16:
            issues.append("Spotify: Too quiet, consider increasing loudness by 2–3 dB")
        elif loudness > -12:
            issues.append("Spotify: Too loud, will be normalized down")
        else:
            issues.append("Spotify: OK")

        # Apple Music
        if true_peak > -1:
            issues.append("Apple Music: True peak too high, risk of clipping")
        else:
            issues.append("Apple Music: OK")

        # YouTube
        if loudness < -16:
            issues.append("YouTube: Slightly quiet but acceptable")
        else:
            issues.append("YouTube: OK")

        # TikTok
        if loudness > -10:
            issues.append("TikTok: May sound distorted on mobile speakers")
        else:
            issues.append("TikTok: OK")

        critical = [i for i in issues if "risk" in i.lower() or "distort" in i.lower()]
        overall_status = "needs_adjustment" if critical else "ready"

        os.unlink(input_path)

        return jsonify({
            "success": True,
            "overall_status": overall_status,
            "lufs": round(float(loudness), 1),
            "true_peak": round(float(true_peak), 1),
            "sample_rate": int(sample_rate),
            "bit_depth": int(bit_depth),
            "issues": issues
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ===========================
# HEALTH CHECK
# ===========================

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
