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

# ---------- basic file config ----------
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/tmp/audio")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def download_file(url: str, suffix: str = ".wav") -> str:
    """
    Download file from remote URL to a temp file path and return the path.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(tmp.name, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tmp.name


def save_and_get_url(audio: np.ndarray, sr: int, filename: str) -> str:
    """
    Save numpy audio array to UPLOAD_FOLDER and return public URL.
    """
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    sf.write(filepath, audio, sr)
    return f"{BASE_URL}/files/{filename}"


@app.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)


# ---------- health ----------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ---------- simple "stem" split (VERY basic, not real AI separation) ----------
@app.route("/api/separate-stems", methods=["POST"])
def separate_stems():
    """
    Very simple fake stem split so the frontend has something to work with
    WITHOUT using Spleeter (which doesn't work on Python 3.13).

    It roughly emphasizes high frequencies as "vocals" and the rest as "instrumental".

    Request JSON:
      { "audio_url": "https://...", "project_id": "optional" }
    """
    try:
        data = request.get_json(force=True)
        audio_url = data.get("audio_url")
        project_id = data.get("project_id") or str(uuid.uuid4())

        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        path = download_file(audio_url, suffix=".wav")
        y, sr = librosa.load(path, sr=44100, mono=True)

        # crude "vocal" emphasis using preemphasis (boost high freq)
        vocals = librosa.effects.preemphasis(y, coef=0.97)
        # rough "instrumental" as residual
        instrumental = y - 0.6 * vocals

        # normalize a bit
        vocals = vocals / (np.max(np.abs(vocals)) + 1e-9) * 0.95
        instrumental = instrumental / (np.max(np.abs(instrumental)) + 1e-9) * 0.95

        vocals_url = save_and_get_url(vocals, sr, f"{project_id}_vocals.wav")
        instrumental_url = save_and_get_url(instrumental, sr, f"{project_id}_instrumental.wav")

        os.unlink(path)

        return jsonify({
            "success": True,
            "vocals_url": vocals_url,
            "instrumental_url": instrumental_url
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------- key & bpm detection ----------
@app.route("/api/analyze", methods=["POST"])
def analyze_audio():
    """
    Detect BPM and musical key from an audio file.
    Request JSON:
      { "audio_url": "https://..." }
    """
    try:
        data = request.get_json(force=True)
        audio_url = data.get("audio_url")
        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        path = download_file(audio_url, suffix=".wav")
        y, sr = librosa.load(path, sr=22050, mono=True)

        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, (list, np.ndarray)):
            bpm = float(tempo[0])
        else:
            bpm = float(tempo)

        # KEY (simple estimate using chroma and Krumhansl profiles)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = chroma.mean(axis=1)

        key_names = ["C", "C#", "D", "D#", "E", "F",
                     "F#", "G", "G#", "A", "A#", "B"]

        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
             2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
             2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        best_key = None
        best_mode = None
        best_corr = -1.0

        for i in range(12):
            major_corr = np.corrcoef(chroma_avg, np.roll(major_profile, i))[0, 1]
            minor_corr = np.corrcoef(chroma_avg, np.roll(minor_profile, i))[0, 1]

            if major_corr > best_corr:
                best_corr = major_corr
                best_key = key_names[i]
                best_mode = "major"

            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = key_names[i]
                best_mode = "minor"

        key = f"{best_key} {best_mode}"

        os.unlink(path)

        return jsonify({"success": True, "bpm": round(bpm), "key": key})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------- simple autotune / pitch correction ----------
@app.route("/api/autotune", methods=["POST"])
def autotune_vocals():
    """
    Very simple pitch-correction style effect.
    This is not pro-level Autotune but gives a tuned / processed sound.
    Request JSON:
      {
        "audio_url": "https://...",
        "key": "C major",        # optional, default C major
        "style": "natural|medium|hard",
        "project_id": "optional-id"
      }
    """
    try:
        data = request.get_json(force=True)
        audio_url = data.get("audio_url")
        key = (data.get("key") or "C major").strip()
        style = (data.get("style") or "medium").lower()
        project_id = data.get("project_id") or str(uuid.uuid4())

        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        if style not in ["natural", "medium", "hard"]:
            style = "medium"

        path = download_file(audio_url, suffix=".wav")
        y, sr = librosa.load(path, sr=44100, mono=True)

        # estimate fundamental frequency contour
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )

        # build scale from key
        key_root = key.split()[0].upper()
        is_minor = "MINOR" in key.upper()

        key_to_midi = {
            "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
            "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
            "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11
        }
        root = key_to_midi.get(key_root, 0)

        major_scale = [0, 2, 4, 5, 7, 9, 11]
        minor_scale = [0, 2, 3, 5, 7, 8, 10]
        scale = minor_scale if is_minor else major_scale

        # All notes in all octaves for the scale
        scale_midi = []
        for octave in range(0, 9):
            for interval in scale:
                scale_midi.append(root + interval + 12 * octave)
        scale_midi = np.array(scale_midi)

        # convert f0 to midi
        f0_midi = librosa.hz_to_midi(f0, offset=0.0)
        target_f0 = f0.copy()

        # snap each voiced frame to nearest scale note
        for i, (freq, voiced) in enumerate(zip(f0_midi, voiced_flag)):
            if not voiced or np.isnan(freq):
                continue
            nearest = scale_midi[np.argmin(np.abs(scale_midi - freq))]
            target_f0[i] = librosa.midi_to_hz(nearest)

        # compute pitch-shift in semitones frame-wise
        # we will use a single global shift as a simplification
        valid = (~np.isnan(f0)) & (~np.isnan(target_f0)) & (voiced_flag)
        if np.any(valid):
            avg_shift_semitones = np.median(
                librosa.hz_to_midi(target_f0[valid]) - librosa.hz_to_midi(f0[valid])
            )
        else:
            avg_shift_semitones = 0.0

        # style strength
        if style == "natural":
            shift = avg_shift_semitones * 0.4
        elif style == "medium":
            shift = avg_shift_semitones * 0.7
        else:  # hard
            shift = avg_shift_semitones

        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)

        # slight dynamic range control
        y_out = y_shifted / (np.max(np.abs(y_shifted)) + 1e-9) * 0.95

        tuned_url = save_and_get_url(y_out, sr, f"{project_id}_tuned.wav")

        os.unlink(path)

        return jsonify({"success": True, "tuned_url": tuned_url})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------- mix & master ----------
@app.route("/api/mix-master", methods=["POST"])
def mix_master():
    """
    Mix and master a vocal file on top of an instrumental.
    Request JSON:
      {
        "vocal_url": "https://...",
        "instrumental_url": "https://...",
        "project_id": "optional-id"
      }
    """
    try:
        data = request.get_json(force=True)
        vocal_url = data.get("vocal_url")
        instrumental_url = data.get("instrumental_url")
        project_id = data.get("project_id") or str(uuid.uuid4())

        if not vocal_url or not instrumental_url:
            return jsonify({"success": False, "error": "vocal_url and instrumental_url are required"}), 400

        vocal_path = download_file(vocal_url, suffix=".wav")
        instrumental_path = download_file(instrumental_url, suffix=".wav")

        vocals, sr = librosa.load(vocal_path, sr=44100, mono=True)
        instrumental, _ = librosa.load(instrumental_path, sr=44100, mono=True)

        # pad to same length
        max_len = max(len(vocals), len(instrumental))
        if len(vocals) < max_len:
            vocals = np.pad(vocals, (0, max_len - len(vocals)))
        if len(instrumental) < max_len:
            instrumental = np.pad(instrumental, (0, max_len - len(instrumental)))

        # simple mixing: high-pass-ish emphasis on vocals
        vocals_pre = librosa.effects.preemphasis(vocals, coef=0.97)
        vocals_enhanced = vocals_pre + 0.1 * librosa.effects.harmonic(vocals_pre)

        mix = instrumental * 0.85 + vocals_enhanced * 0.9

        # soft clip / limiter
        peak = np.max(np.abs(mix)) + 1e-9
        target_peak = 10 ** (-1.0 / 20)  # -1 dB
        mix_norm = mix * (target_peak / peak)
        limit = 10 ** (-0.3 / 20)
        mix_limited = np.clip(mix_norm, -limit, limit)

        master_url = save_and_get_url(mix_limited, sr, f"{project_id}_master.wav")

        os.unlink(vocal_path)
        os.unlink(instrumental_path)

        return jsonify({"success": True, "master_url": master_url})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------- platform check ----------
@app.route("/api/platform-check", methods=["POST"])
def platform_check():
    """
    Check loudness / peaks & give feedback for platforms.
    Request JSON:
      { "audio_url": "https://..." }
    """
    try:
        data = request.get_json(force=True)
        audio_url = data.get("audio_url")
        if not audio_url:
            return jsonify({"success": False, "error": "audio_url is required"}), 400

        path = download_file(audio_url, suffix=".wav")

        y, sr = sf.read(path, always_2d=True)
        # y: shape (n_samples, n_channels)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)

        true_peak = 20 * np.log10(np.max(np.abs(y)) + 1e-10)

        info = sf.info(path)
        sample_rate = info.samplerate
        # naive bit depth guess
        subtype = str(info.subtype).lower()
        if "24" in subtype:
            bit_depth = 24
        elif "32" in subtype:
            bit_depth = 32
        else:
            bit_depth = 16

        issues = []

        # Spotify target roughly -14 LUFS / -1 dBTP
        if loudness < -16:
            issues.append("Spotify: Track is a bit quiet (under -16 LUFS).")
        elif loudness > -12:
            issues.append("Spotify: Track is quite loud and will be turned down.")
        else:
            issues.append("Spotify: Loudness is in a good range.")

        if true_peak > -1:
            issues.append("Apple Music: True peak higher than -1 dB, risk of clipping.")
        else:
            issues.append("Apple Music: True peak is safe.")

        if loudness > -13:
            issues.append("YouTube: Louder than -13 LUFS, will be normalized down.")
        else:
            issues.append("YouTube: Loudness is fine.")

        if loudness > -10:
            issues.append("TikTok: Might sound a bit squashed on small speakers.")
        else:
            issues.append("TikTok: Should translate well on mobile.")

        critical = [i for i in issues if "risk" in i.lower() or "squashed" in i.lower()]
        overall_status = "needs_adjustment" if critical else "ready"

        os.unlink(path)

        return jsonify({
            "success": True,
            "overall_status": overall_status,
            "lufs": round(float(loudness), 1),
            "true_peak": round(float(true_peak), 1),
            "sample_rate": sample_rate,
            "bit_depth": bit_depth,
            "issues": issues,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
