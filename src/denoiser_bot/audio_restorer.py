import librosa
import numpy as np
from src.denoiser_bot.config import *
import soundfile as sf

class AudioRestorer:

    def revert_features_to_audio(self, features, phase, mean, std):
        """
        Function for scaling the outpus back to the original range
        :param features: outputs from neural network in numpy array
        :param phase: saved at first features preparation stage phase value
        :param mean: saved at first features preparation stage mean value
        :param std: saved at first features preparation stage std value
        :return: numpy array with restored audio
        """
        if mean and std:
            features = std * features + mean
            
        phase = np.transpose(phase, (1, 0))

        features = np.squeeze(features)
        features = features * np.exp(1j * phase)
        features = np.transpose(features, (1, 0))
        
        restored = self._restore_audio(features)
        return restored


    def _restore_audio(self, stft_features):
        """
        Function for restoring audio from stft spectrogram
        :param stft_features: outputs from neural network in numpy array
        :return: numpy array with restored audio
        """
        return librosa.istft(stft_features, win_length=WINDOW_LENGTH, hop_length=OVERLAP, window='hamming', center=True)


    def write_audio(self, denoised, filename):
        """
        Write audiofile
        :param denoised: numpy array with restored audio
        :param filename: str filename
        :return: path to file in str
        """
        path = f'audios/{filename}.wav'
        sf.write(path, denoised, SAMPLE_RATE)
        return path
