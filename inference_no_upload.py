from flask import Flask, request
from TTS.api import TTS
import os
import librosa
import soundfile as sf
import torch
import argparse
import jsonify

voices_dir = 'user_voices'
if not os.path.exists(voices_dir):
    os.makedirs(voices_dir)
output_dir = 'output_voices'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ip = '54.91.170.78'

wav_freq = 16000
cuda = torch.cuda.is_available()
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=cuda)
print('Running inference on {}'.format('cpu' if not cuda else 'cuda'))

app = Flask(__name__)


def preprocess_audio(wav_file, freq=16000):
    y, sr = librosa.load(wav_file, sr=freq)
    sf.write(wav_file, y, sr)
    return wav_file


@app.route('/clone-voice', methods=['POST'])
def clone_voice_api():

    if 'speaker_wav' not in request.form:
        return 'Error: No WAV file provided'
    if 'text' not in request.form:
        return 'Error: No text string provided'
    if 'character' not in request.form:
        return 'Error: No character name provided'

    # get the text string from the request
    text = request.form['text']
    char_name = request.form['character']
    speaker_wav = request.form['speaker_wav']  # path to wav
    print('Character: {}\nSpeaker voice: {}\nText: {}'.format(char_name, speaker_wav, text))

    print('Resampling WAV to {} frequency'.format(wav_freq))
    speaker_wav = preprocess_audio(speaker_wav, freq=wav_freq)  # resample WAV

    # generate the TTS audio file 
    output_wav = os.path.join(output_dir, f'{char_name}.wav')
    tts.tts_to_file(text, speaker_wav=speaker_wav, language='en', file_path=output_wav)

    # return the TTS audio file as a binary response
    '''with open(output_wav, 'rb') as f:
        audio_data = f.read()
    return audio_data, {'Content-Type': 'audio/wav'}'''

    # Return a URL pointing to the audio file
    audio_url = f'http://{ip}/{char_name}.wav'
    return jsonify({'audio_url': audio_url})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default='5000')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)


