import os
import cv2
import moviepy.editor as mp
from audio_animation.deepspeech.voice_processor import VoiceProcessor


class Dataset:
    @staticmethod
    def generate(video_folder, save_folder, cuda=True):
        _, _, video_names = os.walk(video_folder)
        voice_processor = VoiceProcessor()
        for video_name in video_names:
            video_with_audio = mp.VideoFileClip(video_folder + '/' + video_name)
            video_with_audio.audio.write_audiofile(save_folder + '/' + "temp_audio.wav")
            audio_features = voice_processor.execute(save_folder + '/' + "temp_audio.wav")
            video = cv2.VideoCapture(video_folder_path + '/' + video_name)

