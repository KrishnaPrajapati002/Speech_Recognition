from flask import Flask, request, jsonify, render_template
import numpy as np
import io
import subprocess
from speech import MultiLanguageSpeechRecognizer
import soundfile as sf

app = Flask(__name__)
recognizer = MultiLanguageSpeechRecognizer()

@app.before_request
def log_all_requests():
    print(f"{request.method}{request.path}")

@app.route('/')
def index():
    return render_template('index.html', languages=recognizer.get_supported_languages())

def convert_audio_to_wav_bytes(audio_bytes):
    # Use ffmpeg to convert to WAV PCM 16-bit LE
    process = subprocess.Popen(
        ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', 'pipe:1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = process.communicate(input=audio_bytes)

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {err.decode()}")

    return out

@app.route("/transcribe", methods=['POST'])
def transcribe():
    try:
        audio_data = request.files['audio']
        language = request.form.get('language', None)

        raw_audio = audio_data.read()
        wav_bytes = convert_audio_to_wav_bytes(raw_audio)

        # Convert audio to numpy array
        audio_bytes = audio_data.read()
        audio_np, sample_rate = sf.read(io.BytesIO(wav_bytes))

        if len(audio_np.shape) > 1:
            audio_np = audio_np[:, 0]  # Use first channel

        # Resample if needed
        if sample_rate != recognizer.sample_rate:
            audio_np = np.interp(
                np.linspace(0, len(audio_np), int(len(audio_np) * recognizer.sample_rate / sample_rate)),
                np.arange(len(audio_np)),
                audio_np
            )

        # Run transcription
        text, detected_lang = recognizer.transcribe_audio_multilingual(audio_np, force_language=language or None)

        return jsonify({
            'transcription': text,
            'language': detected_lang
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
