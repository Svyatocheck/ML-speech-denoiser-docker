from prepare_audio import FeatureInputGenerator
from restore_audio import AudioRestorer

from flask import Flask, request, send_file, render_template
import os
import moviepy.editor as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

AUDIOS_PATH = 'audios'
VIDEOS_PATH = 'videos'

# Get model
model = tf.keras.models.load_model('speech_model.h5')

app = Flask('speech-denoising')

@app.route('/')  
def main():  
    return render_template("index.html")  

@app.route('/send_audio', methods=['POST'])
def receive_and_denoise():
    file = request.files['file']
    
    if file.filename.endswith('.mp3') or file.filename.endswith('.wav'):
        filepath = f"{AUDIOS_PATH}/{file.filename}"
        file.save(filepath)
        
        clean_audio = clean(filepath)
        
        if os.path.isfile(clean_audio): 
            response = send_file(clean_audio, mimetype="audio/wav", as_attachment=True)
            os.remove(filepath)
            os.remove(clean_audio)
            return response
        
    return render_template("error.html")
    # return {"message": "cannot download audio"}, 401


@app.route('/send_video', methods=['POST'])
def video_denoise():
    file = request.files['file']
    if file.filename.endswith('.mp4') or file.filename.endswith('.mov'):
        clean_filename = get_clean_filename(file.filename)
        video_path = f'{VIDEOS_PATH}/{file.filename}'
        file.save(video_path)
        
        video = mp.VideoFileClip(video_path)
        
        audio_path = f"{AUDIOS_PATH}/{clean_filename}.wav"
        video.audio.write_audiofile(audio_path)
        
        clean_audio_path = clean(audio_path, True)
        clip_audio = mp.AudioFileClip(clean_audio_path)
        
        ready_video = video.without_audio()
        ready_video.audio = clip_audio
        
        denoised_video_path = f"{VIDEOS_PATH}/{clean_filename}_denoised.mp4"
        ready_video.write_videofile(denoised_video_path)
        
        if os.path.isfile(denoised_video_path):
            response = send_file(denoised_video_path, as_attachment=True)
            os.remove(video_path)
            os.remove(audio_path)
            os.remove(clean_audio_path)
            os.remove(denoised_video_path)
            return response
    return render_template("error.html")
    # return {"message": "cannot download video"}, 401


def get_clean_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def clean(filepath : str, isVideo = False):
    try:    
        input_generator = FeatureInputGenerator()
        prepared_features = input_generator.start_preprocess(filepath, isVideo)
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