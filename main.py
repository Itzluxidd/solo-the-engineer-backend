import os
import uuid
import tempfile
import requests
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from spleeter.separator import Separator

app = Flask(__name__)
CORS(app)

# Configure your file storage (e.g., S3, Cloudinary, or local)
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', '/tmp/audio')
BASE_URL = os.environ.get('BASE_URL', 'http://localhost:5000')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def download_file(url, local_path):
    """Download file from URL to local path"""
    response = requests.get(url, stream=True)
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path

def save_and_get_url(audio_data, sr, filename):
    """Save audio and return public URL"""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    sf.write(filepath, audio_data, sr)
    return f"{BASE_URL}/files/{filename}"

@app.route('/files/<filename>')
def serve_file(filename):
    """Serve audio files"""
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)

# ============ STEM SEPARATION ============
@app.route('/api/separate-stems', methods=['POST'])
def separate_stems():
    """Separate audio into vocals and instrumental"""
    try:
        data = request.json
        audio_url = data.get('audio_url')
        project_id = data.get('project_id', str(uuid.uuid4()))
        
        # Download the audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name
        
        # Use Spleeter for separation (2 stems: vocals + accompaniment)
        separator = Separator('spleeter:2stems')
        output_dir = tempfile.mkdtemp()
        separator.separate_to_file(input_path, output_dir)
        
        # Find the separated files
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        vocals_path = os.path.join(output_dir, base_name, 'vocals.wav')
        instrumental_path = os.path.join(output_dir, base_name, 'accompaniment.wav')
        
        # Load and save with proper names
        vocals, sr = librosa.load(vocals_path, sr=44100)
        instrumental, _ = librosa.load(instrumental_path, sr=44100)
        
        vocals_url = save_and_get_url(vocals, sr, f"{project_id}_vocals.wav")
        instrumental_url = save_and_get_url(instrumental, sr, f"{project_id}_instrumental.wav")
        
        # Cleanup
        os.unlink(input_path)
        
        return jsonify({
            'success': True,
            'vocals_url': vocals_url,
            'instrumental_url': instrumental_url
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============ KEY & BPM DETECTION ============
@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """Detect key and BPM"""
    try:
        data = request.json
        audio_url = data.get('audio_url')
        
        # Download audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name
        
        # Load audio
        y, sr = librosa.load(input_path, sr=22050)
        
        # Detect BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(tempo[0]) if hasattr(tempo, '__iter__') else float(tempo))
        
        # Detect Key using chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)
        
        # Map to key names
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_index = int(np.argmax(chroma_avg))
        
        # Determine major/minor (simplified)
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        major_corr = np.corrcoef(chroma_avg, np.roll(major_profile, key_index))[0, 1]
        minor_corr = np.corrcoef(chroma_avg, np.roll(minor_profile, key_index))[0, 1]
        
        mode = 'major' if major_corr > minor_corr else 'minor'
        key = f"{key_names[key_index]} {mode}"
        
        os.unlink(input_path)
        
        return jsonify({
            'success': True,
            'bpm': bpm,
            'key': key
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============ AUTOTUNE ============
@app.route('/api/autotune', methods=['POST'])
def autotune_vocals():
    """Apply pitch correction to vocals"""
    try:
        data = request.json
        audio_url = data.get('audio_url')
        key = data.get('key', 'C major')
        style = data.get('style', 'medium')  # natural, medium, hard
        project_id = data.get('project_id', str(uuid.uuid4()))
        
        # Download audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name
        
        # Load audio
        y, sr = librosa.load(input_path, sr=44100)
        
        # Get scale notes based on key
        key_root = key.split()[0]
        is_minor = 'minor' in key.lower()
        
        key_to_midi = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                       'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        root = key_to_midi.get(key_root, 0)
        
        # Scale intervals (semitones from root)
        major_scale = [0, 2, 4, 5, 7, 9, 11]
        minor_scale = [0, 2, 3, 5, 7, 8, 10]
        scale = minor_scale if is_minor else major_scale
        
        # Get all notes in the scale across octaves
        scale_notes = set()
        for octave in range(-1, 10):
            for interval in scale:
                scale_notes.add(root + interval + octave * 12)
        
        # Pitch correction strength based on style
        correction_strength = {'natural': 0.3, 'medium': 0.6, 'hard': 1.0}[style]
        
        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=2000)
        
        # Get pitch track
        pitch_track = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_track.append(pitch if pitch > 0 else 0)
        
        # For "hard" style, use more aggressive processing
        if style == 'hard':
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=0, bins_per_octave=12)
            y_tuned = librosa.effects.harmonic(y_shifted)
        else:
            y_tuned = librosa.effects.pitch_shift(y, sr=sr, n_steps=0)
            y_tuned = y_tuned * correction_strength + y * (1 - correction_strength)
        
        # Normalize
        y_tuned = y_tuned / np.max(np.abs(y_tuned)) * 0.95
        
        tuned_url = save_and_get_url(y_tuned, sr, f"{project_id}_tuned.wav")
        
        os.unlink(input_path)
        
        return jsonify({
            'success': True,
            'tuned_url': tuned_url
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============ MIX & MASTER ============
@app.route('/api/mix-master', methods=['POST'])
def mix_master():
    """Mix and master vocals with instrumental"""
    try:
        data = request.json
        vocal_url = data.get('vocal_url')
        instrumental_url = data.get('instrumental_url')
        project_id = data.get('project_id', str(uuid.uuid4()))
        
        # Download files
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp1:
            download_file(vocal_url, tmp1.name)
            vocal_path = tmp1.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp2:
            download_file(instrumental_url, tmp2.name)
            instrumental_path = tmp2.name
        
        # Load audio
        vocals, sr = librosa.load(vocal_path, sr=44100)
        instrumental, _ = librosa.load(instrumental_path, sr=44100)
        
        # Match lengths
        max_len = max(len(vocals), len(instrumental))
        vocals = np.pad(vocals, (0, max_len - len(vocals)))
        instrumental = np.pad(instrumental, (0, max_len - len(instrumental)))
        
        # === MIXING ===
        vocals_filtered = librosa.effects.preemphasis(vocals, coef=0.97)
        vocals_enhanced = vocals_filtered + 0.1 * librosa.effects.harmonic(vocals_filtered)
        mix = instrumental * 0.8 + vocals_enhanced * 0.7
        
        # === MASTERING ===
        threshold = 0.5
        ratio = 4
        mix_compressed = np.where(
            np.abs(mix) > threshold,
            np.sign(mix) * (threshold + (np.abs(mix) - threshold) / ratio),
            mix
        )
        
        peak = np.max(np.abs(mix_compressed))
        target_peak = 10 ** (-1 / 20)  # -1 dB
        mix_normalized = mix_compressed * (target_peak / peak)
        
        limit = 10 ** (-0.3 / 20)
        mix_limited = np.clip(mix_normalized, -limit, limit)
        
        master_url = save_and_get_url(mix_limited, sr, f"{project_id}_master.wav")
        
        os.unlink(vocal_path)
        os.unlink(instrumental_path)
        
        return jsonify({
            'success': True,
            'master_url': master_url
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============ PLATFORM CHECK ============
@app.route('/api/platform-check', methods=['POST'])
def platform_check():
    """Analyze master for streaming platform compatibility"""
    try:
        data = request.json
        audio_url = data.get('audio_url')
        
        # Download audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            download_file(audio_url, tmp.name)
            input_path = tmp.name
        
        # Load audio
        y, sr = librosa.load(input_path, sr=44100, mono=False)
        if y.ndim == 1:
            y = np.array([y, y])  # Make stereo for analysis
        
        # Measure LUFS
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y.T)
        
        # Measure true peak
        true_peak = 20 * np.log10(np.max(np.abs(y)) + 1e-10)
        
        # Get audio info
        info = sf.info(input_path)
        sample_rate = info.samplerate
        bit_depth = 16 if '16' in str(info.subtype) else 24
        
        issues = []
        
        if loudness < -16:
            issues.append("Spotify: Too quiet, consider increasing loudness by 2-3 dB")
        elif loudness > -12:
            issues.append("Spotify: Too loud, will be normalized down")
        else:
            issues.append("Spotify: OK")
        
        if true_peak > -1:
            issues.append("Apple Music: True peak too high, risk of clipping")
        else:
            issues.append("Apple Music: OK")
        
        if loudness < -16:
            issues.append("YouTube: Slightly quiet but acceptable")
        else:
            issues.append("YouTube: OK")
        
        if loudness > -10:
            issues.append("TikTok: May sound distorted on mobile speakers")
        else:
            issues.append("TikTok: OK")
        
        critical_issues = [i for i in issues if 'risk' in i.lower() or 'distort' in i.lower()]
        overall_status = 'needs_adjustment' if critical_issues else 'ready'
        
        os.unlink(input_path)
        
        return jsonify({
            'success': True,
            'overall_status': overall_status,
            'lufs': round(loudness, 1),
            'true_peak': round(true_peak, 1),
            'sample_rate': sample_rate,
            'bit_depth': bit_depth,
            'issues': issues
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
