SAMPLE_RATE = 16000

WINDOW_LENGTH = 512

OVERLAP = round(0.5 * WINDOW_LENGTH)  # 50%

N_FFT = WINDOW_LENGTH

N_FEATURES = N_FFT // 2 + 1  # 257

N_SEGMENTS = 1

audio_extensions = [".wav", ".mp3", ".oga"]

video_extensions = [".mp4", ".wmv"]

audio_types = ["audio", "voice"]

greetingsEnMsg = "Hello, I'm a telegram bot for cleaning up speech recordings from background noise. I can help you to reduce noise in audio or video files so that you can hear the speaker clearly. However, I'm just learning how to do it right and so I can sometimes make mistakes, but I promise to get better. I hope you find my services helpful!\nLet's start with choosing a file format."
