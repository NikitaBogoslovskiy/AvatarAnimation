import math

from progress.bar import Bar
import cv2
from animation.concurrent.video_animation import VideoAnimationParams, video_animation_pipeline
from animation.concurrent.audio_animation import AudioAnimationParams, audio_animation_pipeline
from audio_animation.wav2vec2.voice_processor import VoiceProcessor
import moviepy.editor as mp
import os
from utils.audio_functions import adapt_audio_to_frame_rate
import torch
from multiprocessing import Process, Queue
from FLAME.utils import upload_upper_masks
from utils.video_functions import add_audio_to_video
from video_animation.visualizer.offline_visualizer import OfflineVisualizer
import numpy as np
from config.paths import PROJECT_DIR


class Animation:
    def __init__(self,
                 cuda=True,
                 audio_support_level=0.5):
        self.cuda = cuda
        self.audio_support_level = audio_support_level
        self.video_animation_params = None
        self.audio_animation_params = None
        self.voice_processor = VoiceProcessor()
        self.masks = upload_upper_masks()
        self.upper_face_indices = [*self.masks.forehead, *self.masks.eye_region, *self.masks.nose]
        self.flame_surfaces = np.load(f"{PROJECT_DIR}/FLAME/model/surfaces.npy")
        self.output_video_path_without_audio = None
        self.output_video_path_with_audio = None
        self.input_audio_path = None
        self.visualizer = None
        self.frames_number = None

    def set_parameters(self, video_path, neutral_face_path=None):
        self.video_animation_params = VideoAnimationParams(cuda=self.cuda,
                                                           video_path=video_path,
                                                           neutral_face_path=neutral_face_path)
        self.audio_animation_params = AudioAnimationParams(cuda=self.cuda)
        directory = os.path.dirname(self.video_animation_params.video_path)
        video_name = os.path.splitext(os.path.basename(self.video_animation_params.video_path))[0]
        self.output_video_path_without_audio = f"{directory}/output_video_stream.mp4"
        self.output_video_path_with_audio = f"{directory}/{video_name}_output.mp4"
        self.input_audio_path = directory + '/' + "temp_audio.wav"
        input_video_cv2 = cv2.VideoCapture(video_path)
        self.frames_number = int(input_video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
        input_video_cv2.release()
        if neutral_face_path is None:
            self.frames_number -= 1

    def _init_visualizer(self, width, height, frame_rate):
        self.visualizer = OfflineVisualizer(self.output_video_path_without_audio)
        self.visualizer.set_surfaces(self.flame_surfaces)
        self.visualizer.init_settings(animation_resolution=(height, height),
                                      input_resolution=(width, height),
                                      frame_rate=frame_rate)

    def _release_visualizer(self):
        self.visualizer.release()
        add_audio_to_video(input_video_path=self.output_video_path_without_audio, audio_path=self.input_audio_path, output_video_path=self.output_video_path_with_audio)
        os.remove(self.output_video_path_without_audio)
        os.remove(self.input_audio_path)
        print(f"Output video has been saved to '{self.output_video_path_with_audio}'")

    @staticmethod
    def _get_audio_coefficients(blank_features, video_fps):
        blank_features_length = len(blank_features)
        audio_coefficients = [1.0] * len(blank_features)
        sequence_length = int(video_fps * 0.2)
        window_size = sequence_length * 2
        increasing_sequence = np.linspace(0.0, 1.0, sequence_length).tolist()
        descending_sequence = increasing_sequence[::-1]
        blank_start_idx = 0
        blank_series = True
        for blank_idx in range(blank_features_length + 1):
            if blank_idx == blank_features_length or not blank_features[blank_idx]:
                if blank_series:
                    audio_coefficients[blank_start_idx: blank_idx] = [0.0] * (blank_idx - blank_start_idx)
                    if blank_idx - blank_start_idx >= window_size:
                        if blank_start_idx != 0:
                            audio_coefficients[blank_start_idx: blank_start_idx + sequence_length] = descending_sequence
                        if blank_idx != blank_features_length:
                            audio_coefficients[blank_idx - sequence_length: blank_idx] = increasing_sequence
                    else:
                        blank_series_length = blank_idx - blank_start_idx
                        clipped_sequence_length = int(math.ceil(blank_series_length / 2))
                        if blank_start_idx != 0:
                            audio_coefficients[blank_start_idx: blank_start_idx + clipped_sequence_length] = descending_sequence[:clipped_sequence_length]
                        if blank_idx != blank_features_length:
                            audio_coefficients[blank_idx - clipped_sequence_length: blank_idx] = increasing_sequence[sequence_length - clipped_sequence_length:]
                    blank_series = False
            else:
                if not blank_series:
                    blank_start_idx = blank_idx
                    blank_series = True
        return audio_coefficients

    def animate_mesh(self):
        print("Preparing for animation... ", end='')
        input_video = mp.VideoFileClip(self.video_animation_params.video_path)
        video_fps = int(round(input_video.fps))
        self._init_visualizer(int(input_video.w), int(input_video.h), video_fps)
        input_video.audio.write_audiofile(self.input_audio_path, fps=16000, logger=None)
        audio_features = self.voice_processor.execute(self.input_audio_path)
        self.audio_animation_params.audio_path = self.input_audio_path
        # self.audio_animation_params.audio_features = torch.Tensor(audio_features)
        self.audio_animation_params.target_frame_rate = video_fps
        adapted_audio_features = adapt_audio_to_frame_rate(audio_features[0], self.audio_animation_params.target_frame_rate)
        blank_features = (torch.argmax(adapted_audio_features, dim=1) == 35).tolist()
        audio_coefficients = self._get_audio_coefficients(blank_features, video_fps)
        audio_model_outputs = Queue()
        video_model_outputs = Queue()
        video_animation_process = Process(target=video_animation_pipeline, args=(self.video_animation_params, video_model_outputs,))
        audio_animation_process = Process(target=audio_animation_pipeline, args=(self.audio_animation_params, audio_model_outputs,))
        video_animation_process.start()
        audio_animation_process.start()
        frame_idx = 0
        while True:
            while video_model_outputs.qsize() == 0:
                pass
            while audio_model_outputs.qsize() == 0:
                pass
            video_model_output = video_model_outputs.get()
            if video_model_output == -1:
                break
            audio_model_output = audio_model_outputs.get()
            if audio_model_output == -1:
                break
            if frame_idx == 0:
                print("Done.")
                bar = Bar('Video and audio processing', max=min(self.frames_number, len(blank_features)), check_tty=False)
                bar.start()
            _, video_vertices, frame = video_model_output
            _, audio_vertices = audio_model_output
            current_audio_support_level = self.audio_support_level * audio_coefficients[frame_idx]
            final_vertices = audio_vertices * current_audio_support_level + video_vertices * (1 - current_audio_support_level)
            final_vertices[self.upper_face_indices] = video_vertices[self.upper_face_indices]
            self.visualizer.render(final_vertices, frame)
            frame_idx += 1
            bar.next()
        while video_model_outputs.qsize() != 0:
            video_model_outputs.get()
        while audio_model_outputs.qsize() != 0:
            audio_model_outputs.get()
        video_animation_process.join()
        audio_animation_process.join()
        bar.finish()
        self._release_visualizer()
