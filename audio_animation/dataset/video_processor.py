import os
from moviepy.editor import VideoFileClip
import json
from config.paths import PROJECT_DIR


class VideoProcessor:
    @staticmethod
    def split_video_with_intervals(input_video_path, output_directory, interval, start):
        input_video = VideoFileClip(input_video_path)
        video_name = os.path.splitext(os.path.basename(input_video_path))[0]
        duration = input_video.duration
        video_idx = 1
        intervals_number = int((duration - start) // interval)
        for interval_idx in range(intervals_number):
            clip = input_video.subclip(start, start + interval)
            clip.to_videofile(f"{output_directory}/{video_name}_{video_idx}.mp4", codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')
            start += interval
            video_idx += 1

    @staticmethod
    def split_videos_with_intervals(input_directory, output_directory, interval):
        if not os.path.exists(f"{input_directory}/config.json"):
            raise Exception("You need to put config.json to the directory with videos")
        with open(f"{input_directory}/config.json", "r") as f:
            videos_info = json.loads(f.read())
        video_names = next(os.walk(input_directory), (None, None, []))[2]
        videos_names = list(filter(lambda x: x.endswith(".mp4") or x.endswith(".MP4"), video_names))
        for video_name in videos_names:
            VideoProcessor.split_video_with_intervals(input_video_path=f"{input_directory}/{video_name}",
                                                      output_directory=output_directory,
                                                      interval=interval,
                                                      start=videos_info[video_name]["start"])


if __name__ == "__main__":
    VideoProcessor.split_videos_with_intervals(input_directory="D:/thesis/dataset/input",
                                               output_directory=f"{PROJECT_DIR}/audio_animation/dataset/raw_data",
                                               interval=8)
