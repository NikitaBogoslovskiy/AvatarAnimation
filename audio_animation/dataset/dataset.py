from config.paths import PROJECT_DIR
import os
import json
import moviepy.editor as mp
from video_animation.video_animation import VideoAnimation
from FLAME.utils import upload_lips_mask
from audio_animation.wav2vec2.voice_processor import VoiceProcessor
from utils.landmarks import MOUTH_LANDMARKS
from time import time
import datetime


class Dataset:
    @staticmethod
    def save(path, lips_positions, lips_landmarks, audio_features):
        data_item = dict()
        data_item["lips_positions"] = lips_positions
        data_item["lips_landmarks"] = lips_landmarks
        data_item["audio_features"] = audio_features
        with open(path, "w") as f:
            f.write(json.dumps(data_item))

    @staticmethod
    def upload(path):
        with open(path, "r") as f:
            data_item = json.loads(f.read())
        if "lips_positions" not in data_item or "lips_landmarks" not in data_item or "audio_features" not in data_item:
            raise Exception("Wrong file. Must contain 'lips_positions', 'lips_landmarks' and 'audio_features' fields")
        return data_item["lips_positions"], data_item["lips_landmarks"], data_item["audio_features"]

    @staticmethod
    def generate(video_folder, save_folder, frames_batch_size=200, cuda=True):
        video_names = next(os.walk(video_folder), (None, None, []))[2]
        video_names = list(filter(lambda x: x.endswith(".mp4") or x.endswith(".MP4"), video_names))
        videos_number = len(video_names)
        voice_processor = VoiceProcessor()
        video_animation = VideoAnimation(cuda=cuda, offline_mode_batch_size=frames_batch_size)
        video_animation.init_settings()
        video_animation.init_concurrent_mode(processes_number=8)
        lips_mask = upload_lips_mask()
        data_item_idx = 1
        time_sum = 0
        for video_name in video_names:
            start_t = time()
            video_with_audio = mp.VideoFileClip(video_folder + '/' + video_name)
            video_with_audio.audio.write_audiofile(save_folder + '/' + "temp_audio.wav", fps=16000)
            audio_features = voice_processor.execute(save_folder + '/' + "temp_audio.wav").tolist()[0]
            video_animation.set_current_video(video_folder + '/' + video_name)
            video_animation.set_current_neutral_face()
            processed_frames = video_animation.process_frames_concurrently()
            lips_positions = []
            lips_landmarks = []
            print(f"Processing video {data_item_idx}/{videos_number} ({video_name})...")
            while True:
                current_batch_size, output_vertices, _ = next(processed_frames)
                if current_batch_size is None:
                    break
                lips_positions.extend(output_vertices[:current_batch_size, lips_mask].tolist())
                lips_landmarks.extend(video_animation.video_model.landmarks_output[:current_batch_size, MOUTH_LANDMARKS].tolist())
            lips_positions_number = len(lips_positions)
            audio_features_number = len(audio_features)
            if lips_positions_number < audio_features_number:
                audio_features = audio_features[:lips_positions_number]
            elif lips_positions_number > audio_features_number:
                lips_positions = lips_positions[:audio_features_number]
                lips_landmarks = lips_landmarks[:audio_features_number]
            Dataset.save(save_folder + f"/{os.path.splitext(os.path.basename(video_name))[0]}.json", lips_positions, lips_landmarks, audio_features)
            print("Done")
            time_sum += time() - start_t
            remaining_time = int((time_sum / data_item_idx) * (videos_number - data_item_idx))
            print(f"Remaining time: {str(datetime.timedelta(seconds=remaining_time))}")
            data_item_idx += 1
        video_animation.release_concurrent_mode()
        os.remove(save_folder + '/' + "temp_audio.wav")


if __name__ == "__main__":
    Dataset.generate(video_folder=f"{PROJECT_DIR}/audio_animation/dataset/raw_data",
                     save_folder=f"{PROJECT_DIR}/audio_animation/dataset/train_data")
