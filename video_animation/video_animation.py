from config.paths import PROJECT_DIR
import torch
from video_animation.detector.detector import Detector
import cv2
from video_animation.model.video_model import VideoModel, VideoModelExecuteParams
from video_animation.visualizer.online_visualizer import OnlineVisualizer
from video_animation.visualizer.offline_visualizer import OfflineVisualizer
from collections import deque
import numpy as np
from utils.landmarks import align_landmarks, divide_landmarks, LEFT_EYE_LANDMARKS, LEFT_EYEBROW_LANDMARKS, \
    RIGHT_EYEBROW_LANDMARKS, RIGHT_EYE_LANDMARKS, MOUTH_LANDMARKS, NOSE_LANDMARKS, JAW_LANDMARKS, divide_landmarks_batch, transform_frame_to_landmarks
from utils.video_settings import VideoMode
import os
from tqdm import tqdm
from threading import Thread
import queue
from multiprocessing import Process, Queue, Manager, Pool
import time


class VideoAnimation:
    def __init__(self, cuda=True, offline_mode_batch_size=50):
        self.video_stream = None
        self.video_mode = None
        self.visualizer = None
        self.neutral_landmarks = None
        self.offline_mode_batch_size = offline_mode_batch_size
        self.detector = Detector()
        self.cuda = cuda
        self.video_model = VideoModel(self.cuda)
        self.video_model.load_model(
            weights_path=f"{PROJECT_DIR}/video_animation/weights/video_model_1_96900_eaUV2Qs70Tlmn0S.pt")
        self.landmarks_number = None
        self.landmarks_sum = np.zeros((68, 2))
        self.landmarks_history = deque()
        self.momentum = 0.0
        self.local_counter = 0
        self.execution_params = VideoModelExecuteParams()
        self.left_eye_indices = torch.tensor(np.array([*LEFT_EYEBROW_LANDMARKS, *LEFT_EYE_LANDMARKS]))
        self.right_eye_indices = torch.tensor(np.array([*RIGHT_EYEBROW_LANDMARKS, *RIGHT_EYE_LANDMARKS]))
        self.nose_mouth_indices = torch.tensor(np.array([*NOSE_LANDMARKS, *MOUTH_LANDMARKS, *JAW_LANDMARKS]))

    def set_video(self, video_path=0):
        if video_path == 0:
            self.video_mode = VideoMode.ONLINE
            self.visualizer = OnlineVisualizer()
            self.video_stream = cv2.VideoCapture(video_path)
            self.video_model.init_for_execution(batch_size=1)
        else:
            self.video_mode = VideoMode.OFFLINE
            if not os.path.isfile(video_path):
                raise FileNotFoundError("File path is incorrect")
            self.video_stream = cv2.VideoCapture(video_path)
            directory = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.visualizer = OfflineVisualizer(f"{directory}/{video_name}_output.mp4")
            width = int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.visualizer.set_resolution(width, height)
            self.video_model.init_for_execution(batch_size=self.offline_mode_batch_size)
        self.visualizer.set_surfaces(self.video_model.flame_model.flamelayer.faces)

    def init_settings(self):
        self.video_mode = VideoMode.OFFLINE
        self.video_model.init_for_execution(batch_size=self.offline_mode_batch_size)

    def set_current_video(self, video_path):
        self.video_stream = cv2.VideoCapture(video_path)

    def capture_neutral_face(self, photo_path=None):
        if self.video_mode == VideoMode.ONLINE:
            neutral_face_image = self._capture_neutral_face_from_video()
        else:
            neutral_face_image = self._capture_neutral_face_from_photo(photo_path)
        self.detector.get_image(neutral_face_image)
        b, rect = self.detector.detect_face()
        if b:
            landmarks = self.detector.detect_landmarks()
            self.neutral_landmarks = align_landmarks(landmarks)
        if self.video_mode == VideoMode.ONLINE:
            cv2.destroyAllWindows()

    def _capture_neutral_face_from_video(self):
        while True:
            ret, frame = self.video_stream.read()
            self.detector.get_image(frame)
            b, rect = self.detector.detect_face()
            if b:
                self.detector.visualize_bounding_box()
            cv2.imshow('Press "Enter" to capture neutral face', self.detector.image)
            if cv2.waitKey(1) == 13:
                return frame

    def _capture_neutral_face_from_photo(self, photo_path):
        if not os.path.isfile(photo_path):
            raise FileNotFoundError("Photo path is incorrect")
        photo = cv2.imread(photo_path)
        return photo

    def _process_frame(self):
        ret, frame = self.video_stream.read()
        self.detector.get_image(frame)
        b, rect = self.detector.detect_face()
        if b:
            landmarks = self.detector.detect_landmarks()
            self.detector.visualize_landmarks()
            landmarks = align_landmarks(landmarks)
            self.landmarks_history.append(landmarks)
            self.landmarks_sum += landmarks
            if self.local_counter == self.landmarks_number:
                left_landmarks = self.landmarks_history.popleft()
                self.landmarks_sum -= left_landmarks
                smoothed_landmarks = self.landmarks_sum / self.landmarks_number
            else:
                smoothed_landmarks = landmarks
                self.local_counter += 1
            left_eye_dir, right_eye_dir, nose_mouth_dir = \
                divide_landmarks(smoothed_landmarks - self.neutral_landmarks)
            left_eye_dir = torch.Tensor(left_eye_dir)
            right_eye_dir = torch.Tensor(right_eye_dir)
            nose_mouth_dir = torch.Tensor(nose_mouth_dir)
            if self.cuda:
                left_eye_dir = left_eye_dir.cuda()
                right_eye_dir = right_eye_dir.cuda()
                nose_mouth_dir = nose_mouth_dir.cuda()
            self.execution_params.left_eye = self.video_model.neutral_landmarks[self.left_eye_indices][:, :2] + left_eye_dir
            self.execution_params.right_eye = self.video_model.neutral_landmarks[self.right_eye_indices][:, :2] + right_eye_dir
            self.execution_params.nose_mouth = self.video_model.neutral_landmarks[self.nose_mouth_indices][:, :2] + nose_mouth_dir
            return frame, self.video_model.execute(self.execution_params)
        return frame, None

    def process_frames_2(self, processes_number=8):
        with Manager() as manager:
            frames_number = int(self.video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            repeated_model_neutral_landmarks = self.video_model.neutral_landmarks[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
            repeated_human_neutral_landmarks = torch.Tensor(self.neutral_landmarks)[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
            if self.cuda:
                repeated_human_neutral_landmarks = repeated_human_neutral_landmarks.cuda()
            repeated_vertices = torch.Tensor(self.video_model.neutral_vertices)[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
            batch_num = frames_number // self.offline_mode_batch_size
            current_batch_size = self.offline_mode_batch_size

            # landmarks = manager.list([torch.Tensor(self.neutral_landmarks)[None, ]] * current_batch_size)
            landmarks_queue = Queue()
            landmarks_list = [torch.Tensor(self.neutral_landmarks)[None, ]] * current_batch_size
            frames_queue = Queue()
            # frames_queue = queue.Queue()
            frames = [None] * current_batch_size
            processes = []
            # indicators = manager.list([False] * processes_number)
            # indicators = [False] * processes_number
            for process_idx in range(processes_number):
                processes.append(Process(target=transform_frame_to_landmarks, args=(process_idx, frames_queue, landmarks_queue, )))
                processes[process_idx].start()

            for batch_idx in range(batch_num + 1):
                if batch_idx == batch_num:
                    current_batch_size = frames_number - batch_num * self.offline_mode_batch_size
                for frame_idx in tqdm(range(current_batch_size)):
                    frames[frame_idx] = self.video_stream.read()[1]
                    frames_queue.put((frame_idx, frames[frame_idx]))
                while landmarks_queue.qsize() != current_batch_size:
                    pass
                s = landmarks_queue.qsize()
                if batch_idx == batch_num:
                    for process_idx in range(processes_number):
                        frames_queue.put(-1)
                # landmarks_dirs = list(landmarks)
                for _ in range(current_batch_size):
                    frame_idx, vertices = landmarks_queue.get()
                    if vertices is not None:
                        landmarks_list[frame_idx] = vertices
                        continue
                    if frame_idx != 0:
                        landmarks_list[frame_idx] = landmarks_list[frame_idx - 1]
                    else:
                        landmarks_list[frame_idx] = torch.Tensor(self.neutral_landmarks)[None,]
                    # if landmarks_dirs[l_idx] is None:
                    #     if l_idx != 0:
                    #         landmarks_dirs[l_idx] = landmarks_dirs[l_idx - 1]
                    #     else:
                    #         landmarks_dirs[l_idx] = torch.Tensor(self.neutral_landmarks)[None, ]
                    #         if self.cuda:
                    #             landmarks_dirs[l_idx] = landmarks_dirs[l_idx].cuda()
                landmarks_tensor = torch.cat(landmarks_list)
                if self.cuda:
                    landmarks_tensor = landmarks_tensor.cuda()
                left_eye_dirs, right_eye_dirs, nose_mouth_dirs = divide_landmarks_batch(landmarks_tensor - repeated_human_neutral_landmarks[:, :, :2])
                if self.cuda:
                    left_eye_dirs = left_eye_dirs.cuda()
                    right_eye_dirs = right_eye_dirs.cuda()
                    nose_mouth_dirs = nose_mouth_dirs.cuda()
                self.execution_params.left_eye = repeated_model_neutral_landmarks[:, self.left_eye_indices][:, :, :2] + left_eye_dirs
                self.execution_params.right_eye = repeated_model_neutral_landmarks[:, self.right_eye_indices][:, :, :2] + right_eye_dirs
                self.execution_params.nose_mouth = repeated_model_neutral_landmarks[:, self.nose_mouth_indices][:, :, :2] + nose_mouth_dirs
                output = self.video_model.execute(self.execution_params)
                repeated_vertices[:, self.video_model.face_mask] = output[:, self.video_model.face_mask]
                yield current_batch_size, repeated_vertices.cpu(), frames

            b = frames_queue.empty()
            for process_idx in range(processes_number):
                processes[process_idx].join()
                # processes[process_idx].close()
            yield None, None, None

    def process_frames(self):
        frames_number = int(self.video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        repeated_neutral_landmarks = self.video_model.neutral_landmarks[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        repeated_vertices = torch.Tensor(self.video_model.neutral_vertices)[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        if self.cuda:
            repeated_vertices = repeated_vertices.cuda()
        batch_num = frames_number // self.offline_mode_batch_size
        current_batch_size = self.offline_mode_batch_size
        landmarks_dirs = [torch.zeros_like(torch.Tensor(self.neutral_landmarks)[None,])] * current_batch_size
        frames = [None] * current_batch_size
        for batch_idx in range(batch_num + 1):
            if batch_idx == batch_num:
                current_batch_size = frames_number - batch_num * self.offline_mode_batch_size
            for frame_idx in tqdm(range(current_batch_size)):
                _, frame = self.video_stream.read()
                frames[frame_idx] = frame
                self.detector.get_image(frame)
                found, rect = self.detector.detect_face()
                if not found:
                    if frame_idx != 0:
                        landmarks_dirs[frame_idx] = landmarks_dirs[frame_idx - 1]
                    else:
                        landmarks_dirs[frame_idx] = torch.zeros_like(self.neutral_landmarks)
                    continue
                landmarks = self.detector.detect_landmarks()
                self.detector.visualize_landmarks()
                landmarks = align_landmarks(landmarks)
                self.landmarks_history.append(landmarks)
                self.landmarks_sum += landmarks
                if self.local_counter == self.landmarks_number:
                    left_landmarks = self.landmarks_history.popleft()
                    self.landmarks_sum -= left_landmarks
                    smoothed_landmarks = self.landmarks_sum / self.landmarks_number
                else:
                    smoothed_landmarks = landmarks
                    self.local_counter += 1
                landmarks_dirs[frame_idx] = torch.Tensor(smoothed_landmarks - self.neutral_landmarks)[None,]

            left_eye_dirs, right_eye_dirs, nose_mouth_dirs = divide_landmarks_batch(torch.cat(landmarks_dirs))
            if self.cuda:
                left_eye_dirs = left_eye_dirs.cuda()
                right_eye_dirs = right_eye_dirs.cuda()
                nose_mouth_dirs = nose_mouth_dirs.cuda()
            self.execution_params.left_eye = repeated_neutral_landmarks[:, self.left_eye_indices][:, :, :2] + left_eye_dirs
            self.execution_params.right_eye = repeated_neutral_landmarks[:, self.right_eye_indices][:, :, :2] + right_eye_dirs
            self.execution_params.nose_mouth = repeated_neutral_landmarks[:, self.nose_mouth_indices][:, :, :2] + nose_mouth_dirs
            output = self.video_model.execute(self.execution_params)
            repeated_vertices[:, self.video_model.face_mask] = output[:, self.video_model.face_mask]
            yield current_batch_size, repeated_vertices.cpu(), frames
        yield None, None, None

    def animate_mesh(self):
        vertices = torch.Tensor(self.video_model.neutral_vertices)
        if self.cuda:
            vertices = vertices.cuda()
        if self.video_mode == VideoMode.ONLINE:
            self.landmarks_number = 2
            while True:
                _, model_output = self._process_frame()
                if model_output is not None:
                    vertices[self.video_model.face_mask, :] = model_output[0, self.video_model.face_mask]
                self.visualizer.render(vertices.cpu().numpy().squeeze(), pause=0.001)
                cv2.imshow('Window for avatar manipulation', self.detector.image)
                if cv2.waitKey(1) == 27:
                    break
        elif self.video_mode == VideoMode.OFFLINE:
            # self.landmarks_number = 3
            # frames_number = int(self.video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            # for _ in tqdm(range(frames_number)):
            #     input_frame, model_output = self._process_frame()
            #     if model_output is not None:
            #         vertices[self.video_model.face_mask, :] = model_output[0, self.video_model.face_mask]
            #     self.visualizer.render(vertices.cpu().numpy().squeeze(), input_frame)
            processed_frames = self.process_frames()
            while True:
                current_batch_size, output_vertices, input_frames = next(processed_frames)
                print("before break")
                if current_batch_size is None:
                    break
                output_vertices = output_vertices.numpy().squeeze()
                for idx in range(current_batch_size):
                    self.visualizer.render(output_vertices[idx], input_frames[idx])

    def stop(self):
        self.visualizer.release()
        self.video_stream.release()
        cv2.destroyAllWindows()
