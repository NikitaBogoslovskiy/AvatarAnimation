import os
import cv2


class Dataset:
    @staticmethod
    def generate(video_folder_path):
        _, _, video_names = os.walk(video_folder_path)
        for video_name in video_names:
            video = cv2.VideoCapture(video_folder_path + '/' + video_name)

