import tensorflow as tf
from src.denoiser_bot.feature_extractor import FeatureExtractor
from src.denoiser_bot.audio_restorer import AudioRestorer
from src.denoiser_bot.config import *
import telebot
import os
import moviepy.editor as mp

# Reduce warning messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Get model
model = tf.keras.models.load_model('src/denoiser_bot/model/speech_model.h5')

# Get bot
bot = telebot.TeleBot('')


@bot.message_handler(commands=['start'])
def start(message):
    """
    Entry point for our bot. Just sends Hi and some additional info. Also creating audio and video directory.
    :param message: object with content of message
    :return: None
    """
    bot.send_message(message.from_user.id, greetingsEnMsg)
    bot.send_message(message.from_user.id,
                     'Send me file or record an audio/video message and I will try to denoise it. Note: now I can work with mp3/wav audio files and mp4/wmv for video.')


@bot.message_handler(content_types=["document", "audio", "voice", "video", "video_note"])
def message_processing(message):
    """
    This function activating when bot receives either a doc, audio, voice, video, video message.
    :param message: object with content of message
    :return: None
    """
    try:
        # Make user know, that we received the file
        bot.send_message(message.chat.id, "Let's see...")

        # If it's document we need to check extension
        if message.content_type == "document":
            file_info = bot.get_file(message.document.file_id)

            file_extension = get_clean_filename(file_info.file_path)[1]

            if file_extension in video_extensions:
                video_processing(message)

            elif file_extension in audio_extensions:
                audio_processing(message)

            else:
                bot.send_message(message.from_user.id,
                                 "Seems like it's not requested file.")
                return

        # If it's voice/video message we just need to send them to right function
        elif message.content_type in audio_types:
            audio_processing(message)
        else:
            video_processing(message)

    except Exception as ex:
        bot.send_message(
            message.chat.id, "Make sure your files are good to denoise. Otherwise it's problem with me.")
        print(ex)


def audio_processing(message):
    """
    Function for processing audio files.
    :param message: object with content of message
    :return: None
    """
    try:
        # Receive file
        if message.content_type == "document":  # wav
            file_info = bot.get_file(message.document.file_id)

        elif message.content_type == "audio":
            file_info = bot.get_file(message.audio.file_id)  # mp3

        elif message.content_type == "voice":
            file_info = bot.get_file(message.voice.file_id)  # voice

        else:
            bot.send_message(message.from_user.id,
                             "Strange file. I can't recognise it format.")
            return

        # Download file
        downloaded_file = bot.download_file(file_info.file_path)
        filepath = f"audios/{file_info.file_id}.mp3"

        # Write file to directory
        with open(filepath, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.send_message(message.from_user.id,
                         "Wait a sec, I will denoise it...")
        bot.send_chat_action(message.chat.id, 'typing')
        # Denoise file
        clean_audio = clean(filepath)
        if clean_audio is None:
            bot.send_message(message.chat.id, "I can't clean your audio :(")
            return

        # Send file if everyting is fine
        if os.path.isfile(clean_audio):
            with open(clean_audio, 'rb') as audio:
                if message.content_type != 'voice':
                    bot.send_audio(message.from_user.id, audio)
                else:
                    bot.send_voice(message.from_user.id, audio)
            os.remove(filepath)
            os.remove(clean_audio)
            print("Sent.")
        else:
            bot.send_message(
                message.chat.id, "Something is wrong with your file. I'm sorry, I can't clean it.")

    except Exception as ex:
        print(ex)
        bot.send_message(message.chat.id, "Something is wrong :(")


def video_processing(message):
    """
    Function for processing video files.
    :param message: object with content of message
    :return: None
    """
    try:
        # Receive file
        if message.content_type == "video":  # just file
            file_info = bot.get_file(message.video.file_id)

        elif message.content_type == "video_note":
            file_info = bot.get_file(
                message.video_note.file_id)  # video circle

        elif message.content_type == "document":
            file_info = bot.get_file(message.document.file_id)  # video circle

        else:
            bot.send_message(message.from_user.id,
                             "Strange file. I can't recognise it format.")
            return

        # Download file
        downloaded_file = bot.download_file(file_info.file_path)
        filepath = f"videos/{file_info.file_id}.mp4"

        # Save file to the directory
        with open(filepath, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.send_message(message.from_user.id,
                         "Wait a sec, I will denoise it...")

        bot.send_chat_action(message.chat.id, 'typing')
        # Get audio from video
        video = mp.VideoFileClip(filepath)
        audio_path = f"audios/{file_info.file_id}.mp3"
        video.audio.write_audiofile(audio_path)

        # Clean audio
        clean_audio_path = clean(audio_path)
        if clean_audio_path is None:
            bot.send_message(message.chat.id, "I can't clean your video :(")
            return

        clip_audio = mp.AudioFileClip(clean_audio_path)

        # Concatenate new audio with video
        ready_video = video.without_audio()
        ready_video.audio = clip_audio

        # Save new video
        denoised_video_path = f"videos/{file_info.file_id}_denoised.mp4"
        ready_video.write_videofile(denoised_video_path)

        # Send file if everyting is fine
        if os.path.isfile(denoised_video_path):
            with open(denoised_video_path, 'rb') as video:
                if message.content_type != 'video_note':
                    bot.send_video(message.from_user.id, video)
                else:
                    bot.send_video_note(message.from_user.id, video)
            os.remove(filepath)
            os.remove(audio_path)
            os.remove(clean_audio_path)
            os.remove(denoised_video_path)
            print("Sent.")
        else:
            bot.send_message(
                message.chat.id, "Something is wrong with your file. I'm sorry, I can't clean it.")

    except Exception as ex:
        bot.send_message(message.chat.id, "Something is wrong :(")
        print(ex)


def clean(filepath: str):
    try:
        # Extract features from audio to use them in model
        input_generator = FeatureExtractor()
        prepared_features = input_generator.start_preprocess(filepath)
    except Exception as ex:
        print(f"There is problem: {ex}")
        print("Exception during audio preparation.")
        return None

    # Use model to clean audio
    results = model.predict(prepared_features)

    # Restore audio from model results
    output_restorer = AudioRestorer()

    # Save restored audio to directory
    restored = output_restorer.revert_features_to_audio(results,
                                                        input_generator.audio_phase,
                                                        input_generator.mean,
                                                        input_generator.std)
    result = output_restorer.write_audio(
        restored, get_clean_filename(filepath)[0])
    return result


def get_clean_filename(filepath):
    """
    Extract clean filename from filepath
    :param filepath: path to file
    :return: filepath (String)
    """
    return os.path.splitext(os.path.basename(filepath))


if __name__ == '__main__':
    try:
        os.mkdir("audios")
        os.mkdir("videos")
    except OSError as error:
        print(error)
        
    # обязательная для работы бота часть
    bot.polling(none_stop=True, interval=0)
