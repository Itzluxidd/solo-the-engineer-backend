import os
import uuid
import tempfile
import requests
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydub import AudioSegment
from spleeter.separator import Separator

app = Flask(__name__)
CORS(app)

# Where processed audio files will be written in the container
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/tmp/audio")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:5000")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def download_file(url: str, local_path: str) -> str:
    """Download a file from a URL to a local path."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return local_path


def save_and_get_url(audio_data: np.ndarray, sr: int, filename: str) -> str:
    """Save a numpy audio array and return a public URL."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Ensure we don't accidentally blow up values > 1.0
    if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
        audio_data = audio_data.astype(np.float32)

    max_val = np.max(np.abs(audio_data)) + 1e-9
    if max_val > 1.0:
        audio_data = audio_data / max_val

    sf.write(filepath, audio_data, sr)
    return f"{BASE_URL}/files/{filename}"


@app.route("/files/<path:filename>")
def serve_file(filename):
    """Serve generated audio files."""
    return send_from_directory(UPLOAD_FOLDER, filename)


# ---------------------------------------------------
# 1. STEM SEPARATION
# ---------------------------------------------------
@app.route("/api/separate-stems", methods=["POST"])
def separate_stems():
    """
    Separate audio into (approx) vocals and instrumental using Spleeter.
    Input JSON:
      {
        "audio_url": "https://...",
        "project_id": "optional-id"
      }
    """
    try:
        data = request.get_json(force=True)
        audio_url = data.get("audio_url")
        project_id = data.get("project_id", str(uuid.uuid4()))

        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        # Download original file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name

        # Run Spleeter (2 stems: vocals + accompaniment)
        separator = Separator("spleeter:2stems")
        output_dir = tempfile.mkdtemp()
        separator.separate_to_file(input_path, output_dir)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        vocals_path = os.path.join(output_dir, base_name, "vocals.wav")
        instrumental_path = os.path.join(output_dir, base_name, "accompaniment.wav")

        # Load with soundfile
        vocals, sr_v = sf.read(vocals_path, always_2d=False)
        instrumental, sr_i = sf.read(instrumental_path, always_2d=False)

        # If sample rates differ, resample instrumental to vocals using pydub
        if sr_v != sr_i:
            inst_seg = AudioSegment.from_file(instrumental_path)
            inst_seg = inst_seg.set_frame_rate(sr_v)
            samples = np.array(inst_seg.get_array_of_samples()).astype(np.float32)
            instrumental = samples / (2**15)
            sr_i = sr_v

        vocals_url = save_and_get_url(vocals, sr_v, f"{project_id}_vocals.wav")
        instrumental_url = save_and_get_url(
            instrumental, sr_v, f"{project_id}_instrumental.wav"
        )

        os.unlink(input_path)

        return jsonify(
            {
                "success": True,
                "vocals_url": vocals_url,
                "instrumental_url": instrumental_url,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------
# 2. KEY & BPM (PLACEHOLDER)
# ---------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze_audio():
    """
    Placeholder for key & BPM.
    Right now we *don't* use librosa to avoid dependency conflicts.
    It just returns null values so the frontend doesn't break.
    """
    try:
        return jsonify(
            {
                "success": True,
                "bpm": None,
                "key": None,
                "note": "Key/BPM analysis disabled on this backend to keep dependencies simple.",
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------
# 3. "AUTOTUNE" (simple processing, no librosa)
# ---------------------------------------------------
@app.route("/api/autotune", methods=["POST"])
def autotune_vocals():
    """
    Simple vocal processing (NOT true pitch correction).
    It:
      - Normalizes volume
      - Adds a slightly different tone depending on style
    Input JSON:
      {
        "audio_url": "https://...",
        "key": "C major" (ignored for now),
        "style": "natural" | "medium" | "hard",
        "project_id": "optional-id"
      }
    """
    try:
        data = request.get_json(force=True)
        audio_url = data.get("audio_url")
        style = data.get("style", "medium").lower()
        project_id = data.get("project_id", str(uuid.uuid4()))

        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name

        # Use pydub for convenience
        seg = AudioSegment.from_file(input_path)

        # Style-based processing
        if style == "hard":
            # More aggressive compression & slight distortion
            seg = seg.normalize().compress_dynamic_range(threshold=-18.0)
            seg = seg + 4  # boost gain
        elif style == "natural":
            # Gentle normalization only
            seg = seg.normalize()
        else:  # medium
            seg = seg.normalize().compress_dynamic_range(threshold=-22.0)

        # Export to numpy
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)

        # Handle stereo by reshaping
        if seg.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        audio = samples / (2**15)

        tuned_url = save_and_get_url(audio, sr, f"{project_id}_tuned.wav")

        os.unlink(input_path)

        return jsonify({"success": True, "tuned_url": tuned_url})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------
# 4. MIX & MASTER
# ---------------------------------------------------
@app.route("/api/mix-master", methods=["POST"])
def mix_master():
    """
    Mix and master vocals + instrumental.
    Input JSON:
      {
        "vocal_url": "...",
        "instrumental_url": "...",
        "project_id": "optional-id"
      }
    """
    try:
        data = request.get_json(force=True)
        vocal_url = data.get("vocal_url")
        instrumental_url = data.get("instrumental_url")
        project_id = data.get("project_id", str(uuid.uuid4()))

        if not vocal_url or not instrumental_url:
            return (
                jsonify(
                    {"success": False, "error": "vocal_url and instrumental_url required"}
                ),
                400,
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1:
            download_file(vocal_url, tmp1.name)
            vocal_path = tmp1.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
            download_file(instrumental_url, tmp2.name)
            instrumental_path = tmp2.name

        vocals, sr_v = sf.read(vocal_path, always_2d=False)
        instrumental, sr_i = sf.read(instrumental_path, always_2d=False)

        # Force mono for simplicity
        if vocals.ndim > 1:
            vocals = vocals.mean(axis=1)
        if instrumental.ndim > 1:
            instrumental = instrumental.mean(axis=1)

        # Resample to match sample rate if needed using pydub
        if sr_v != sr_i:
            inst_seg = AudioSegment.from_file(instrumental_path)
            inst_seg = inst_seg.set_frame_rate(sr_v)
            samples = np.array(inst_seg.get_array_of_samples()).astype(np.float32)
            instrumental = samples / (2**15)
            sr_i = sr_v

        sr = sr_v

        # Match lengths
        max_len = max(len(vocals), len(instrumental))
        vocals = np.pad(vocals, (0, max_len - len(vocals)))
        instrumental = np.pad(instrumental, (0, max_len - len(instrumental)))

        # Basic vocal "presence" – simple high-pass-ish effect
        preemphasis_coeff = 0.97
        vocals_filtered = np.append(vocals[0], vocals[1:] - preemphasis_coeff * vocals[:-1])

        # Slight boost to vocals
        vocals_enhanced = vocals_filtered * 1.1

        # Mix
        mix = instrumental * 0.8 + vocals_enhanced * 0.7

        # Simple compressor
        threshold = 0.5
        ratio = 4.0
        mix_compressed = np.where(
            np.abs(mix) > threshold,
            np.sign(mix) * (threshold + (np.abs(mix) - threshold) / ratio),
            mix,
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


# ---------------------------------------------------
# 5. PLATFORM CHECK
# ---------------------------------------------------
@app.route("/api/platform-check", methods=["POST"])
def platform_check():
    """
    Analyze loudness & peak for platform readiness.
    Input JSON:
      { "audio_url": "..." }
    """
    try:
        data = request.get_json(force=True)
        audio_url = data.get("audio_url")

        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name

        y, sr = sf.read(input_path, always_2d=False)

        # Force stereo for pyloudnorm (samples x channels)
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)

        true_peak = 20 * np.log10(np.max(np.abs(y)) + 1e-10)

        info = sf.info(input_path)
        sample_rate = info.samplerate
        bit_depth = 16 if "16" in str(info.subtype) else 24

        issues = []

        # Spotify
        if loudness < -16:
            issues.append("Spotify: Too quiet, consider +2–3 dB.")
        elif loudness > -12:
            issues.append("Spotify: Too loud, will be normalized down.")
        else:
            issues.append("Spotify: OK")

        # Apple Music
        if true_peak > -1:
            issues.append("Apple Music: True peak too high, risk of clipping.")
        else:
            issues.append("Apple Music: OK")

        # YouTube
        if loudness < -16:
            issues.append("YouTube: Slightly quiet but acceptable.")
        else:
            issues.append("YouTube: OK")

        # TikTok
        if loudness > -10:
            issues.append("TikTok: May sound distorted on small speakers.")
        else:
            issues.append("TikTok: OK")

        critical_issues = [
            i for i in issues if "risk" in i.lower() or "distort" in i.lower()
        ]
        overall_status = "needs_adjustment" if critical_issues else "ready"

        os.unlink(input_path)

        return jsonify(
            {
                "success": True,
                "overall_status": overall_status,
                "lufs": round(float(loudness), 1),
                "true_peak": round(float(true_peak), 1),
                "sample_rate": sample_rate,
                "bit_depth": bit_depth,
                "issues": issues,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------
# Healthcheck
# ---------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
