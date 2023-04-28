from config.paths import PROJECT_DIR
import os
import sys
import json
import moviepy.editor as mp
from video_animation.video_animation import VideoAnimation
from FLAME.utils import upload_lips_mask
sys.path.append(f"{PROJECT_DIR}/audio_animation/deepspeech")
from audio_animation.deepspeech.voice_processor import VoiceProcessor


class Dataset:
    @staticmethod
    def generate(video_folder, save_folder, frames_batch_size=200, cuda=True):
        video_names = next(os.walk(video_folder), (None, None, []))[2]
        if not os.path.isfile(video_folder + '/' + "neutral.jpg"):
            raise IOError("You need to put image with neutral face in video folder and name it 'neutral.jpg'")
        video_names.remove("neutral.jpg")
        video_names = list(filter(lambda x: x.endswith(".mp4"), video_names))
        voice_processor = VoiceProcessor()
        video_animation = VideoAnimation(cuda=cuda, offline_mode_batch_size=frames_batch_size)
        video_animation.init_settings()
        video_animation.capture_neutral_face(video_folder + '/' + "neutral.jpg")
        video_animation.init_concurrent_mode(processes_number=8)
        lips_mask = upload_lips_mask()
        data_item_idx = 1
        for video_name in video_names:
            video_with_audio = mp.VideoFileClip(video_folder + '/' + video_name)
            video_with_audio.audio.write_audiofile(save_folder + '/' + "temp_audio.wav", fps=16000)
            audio_features = voice_processor.execute(save_folder + '/' + "temp_audio.wav").tolist()
            video_animation.set_current_video(video_folder + '/' + video_name)
            processed_frames = video_animation.process_frames_concurrently()
            lips_positions = []
            while True:
                current_batch_size, output_vertices, _ = next(processed_frames)
                if current_batch_size is None:
                    break
                lips_positions.extend(output_vertices[:current_batch_size, lips_mask].tolist())
            data_item = dict()
            data_item["audio_features"] = audio_features
            data_item["lips_positions"] = lips_positions
            with open(save_folder + f"/{data_item_idx}.json", "w") as f:
                f.write(json.dumps(data_item))
            data_item_idx += 1
        video_animation.release_concurrent_mode()
        os.remove(save_folder + '/' + "temp_audio.wav")


if __name__ == "__main__":
    Dataset.generate(video_folder=f"{PROJECT_DIR}/audio_animation/dataset/raw_data",
                     save_folder=f"{PROJECT_DIR}/audio_animation/dataset/train_data")
