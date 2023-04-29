import librosa
import numpy as np
from constants import *

class FeatureInputGenerator:

    def start_preprocess(self, audio):
        try:
            spectrogram = self._make_spectrograms(audio)
        except:
            raise Exception('Audio is empty, we can not help ya!')
        
        self._put_phase(spectrogram)    
        spectrogram = self._calculate_means(spectrogram)
        
        X = self._reshape_predictors(self._prepare_input_features(spectrogram))
        x_predictor = np.asarray(X).astype(np.float32)
        return x_predictor
    
    
    def _read_audio_files(self, path, normalize):
        '''Загрузка, удаление тихих участков из аудио файла, нормализация.'''
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        if normalize:
            audio, _ = librosa.effects.trim(audio)
            div_fac = 1 / np.max(np.abs(audio)) / 3.0
            audio = audio * div_fac
        return audio
    
    
    def _make_spectrograms(self, audio, clean = False):
        '''Создание STFT диаграмм.'''
        audio = self._read_audio_files(audio, clean)
        stft = librosa.stft(y=audio, n_fft=N_FFT,hop_length=OVERLAP, center=True, window='hamming')
        return stft
    
    
    def _calculate_means(self, spectrogram):
        '''Необходимый в предобработке звукового сигнала этап.
        Взят из статьи 1609.'''
        stft_feature = np.abs(spectrogram)
        mean = np.mean(stft_feature)
        std = np.std(stft_feature)
        self.mean = mean
        self.std = std
        return (stft_feature - mean) / std
    
    
    def _prepare_input_features(self, items):
        '''Формирование векторов из STFT диаграмм.'''
        stft_feature = np.concatenate([items[:,0:N_SEGMENTS-1], items], axis=1)
        stft_segments = np.zeros((N_FEATURES, N_SEGMENTS , stft_feature.shape[1] - N_SEGMENTS + 1))
        for index in range(stft_feature.shape[1] - N_SEGMENTS + 1):
            stft_segments[:,:,index] = stft_feature[:,index:index + N_SEGMENTS]
        return stft_segments
    
    
    def _reshape_predictors(self, items):
        '''Решейп векторов в требуемый НС формат.'''
        predictors = np.reshape(items, (items.shape[0], items.shape[1], 1, items.shape[2]))
        predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
        print(predictors.shape)
        return predictors
    
    
    def _put_phase(self, spectrogram):
        self.audio_phase = np.angle(spectrogram)
