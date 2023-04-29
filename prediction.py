from prepare_audio import FeatureInputGenerator
from restore_audio import AudioRestorer

from flask import Flask, request, send_file
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
AUDIOS_PATH = 'audios/'

# Get model
model = tf.keras.models.load_model('speech_model.h5')

app = Flask('speech-denoising')

@app.route('/send_audio', methods=['POST'])
def receive_and_denoise():
    file = request.files['messageFile']
    
    if file.filename.endswith('.mp3') or file.filename.endswith('.wav'):
        filepath = f"{AUDIOS_PATH}/{file.filename}"
        file.save(filepath)
        
        clean_audio = clean(filepath)
        
        if os.path.isfile(clean_audio):
            return send_file(clean_audio, mimetype="audio/wav", as_attachment=True)

    return {"message": "cannot download audio"}, 401


# @app.route('/get_audio', methods=['GET'])
# def get_audio():
#     file = request.args.get('filename')
    
#     filename = f'{AUDIOS_PATH}/{get_clean_filename(file)}_denoised.wav'

#     if os.path.isfile(filename):
#         return send_file(filename, mimetype="audio/wav", as_attachment=True)
#     else:
#         return {"message": "cannot download audio"}, 401


def get_clean_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def clean(filepath : str):
    try:    
        input_generator = FeatureInputGenerator()
        prepared_features = input_generator.start_preprocess(filepath)
    except:
        print("Exception during audio preparation.")
        return
    
    results = model.predict(prepared_features)
    
    output_restorer = AudioRestorer()
    restored = output_restorer.revert_features_to_audio(results, 
                                                        input_generator.audio_phase, 
                                                        input_generator.mean, 
                                                        input_generator.std)
    return output_restorer.write_audio(restored, get_clean_filename(filepath))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)