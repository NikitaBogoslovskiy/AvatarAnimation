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
from multiprocessing import Process, Queue
from progress.bar import Bar


class VideoAnimation:
    def __init__(self, cuda=True, offline_mode_batch_size=200):
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

    def init_concurrent_mode(self, processes_number=8):
        self.processes_number = processes_number
        self.repeated_model_neutral_landmarks = self.video_model.neutral_landmarks[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        self.repeated_human_neutral_landmarks = torch.Tensor(self.neutral_landmarks)[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        if self.cuda:
            self.repeated_human_neutral_landmarks = self.repeated_human_neutral_landmarks.cuda()
        self.repeated_vertices = torch.Tensor(self.video_model.neutral_vertices)[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        self.landmarks_queue = Queue()
        self.landmarks_list = [torch.Tensor(self.neutral_landmarks)[None, ]] * self.offline_mode_batch_size
        self.frames_queue = Queue()
        self.frames = [None] * self.offline_mode_batch_size
        self.processes = []
        for process_idx in range(self.processes_number):
            self.processes.append(Process(target=transform_frame_to_landmarks, args=(self.frames_queue, self.landmarks_queue)))
            self.processes[process_idx].start()

    def release_concurrent_mode(self):
        for process_idx in range(self.processes_number):
            self.frames_queue.put(-1)
        for process_idx in range(self.processes_number):
            self.processes[process_idx].join()

    def init_sequential_mode(self):
        self.repeated_neutral_landmarks = self.video_model.neutral_landmarks[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        self.repeated_vertices = torch.Tensor(self.video_model.neutral_vertices)[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        if self.cuda:
            self.repeated_vertices = self.repeated_vertices.cuda()
        self.landmarks_dirs = [torch.zeros_like(torch.Tensor(self.neutral_landmarks)[None,])] * self.offline_mode_batch_size
        self.frames = [None] * self.offline_mode_batch_size

    def release_sequential_mode(self):
        pass

    def set_current_video(self, video_path):
        self.video_stream = cv2.VideoCapture(video_path)

    def set_current_neutral_face(self):
        while True:
            _, frame = self.video_stream.read()
            self.detector.get_image(frame)
            b, rect = self.detector.detect_face()
            if b:
                landmarks = self.detector.detect_landmarks()
                self.neutral_landmarks = align_landmarks(landmarks)
                break

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

    def process_frame(self):
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

    def process_frames_concurrently(self):
        frames_number = int(self.video_stream.get(cv2.CAP_PROP_FRAME_COUNT)) - int(self.video_stream.get(cv2.CAP_PROP_POS_FRAMES))
        batch_num = frames_number // self.offline_mode_batch_size
        current_batch_size = self.offline_mode_batch_size
        bar = Bar('Video processing', max=frames_number, check_tty=False)
        lacking_frames = []

        for batch_idx in range(batch_num + 1):
            if batch_idx == batch_num:
                current_batch_size = frames_number - batch_num * self.offline_mode_batch_size
            for frame_idx in range(current_batch_size):
                success, frame = self.video_stream.read()
                if not success:
                    lacking_frames.append(frame_idx)
                    self.frames[frame_idx] = None
                    continue
                self.frames[frame_idx] = frame
                self.frames_queue.put((frame_idx, self.frames[frame_idx]))
            for frame_idx in lacking_frames:
                if frame_idx == 0:
                    local_idx = 1
                    while self.frames[local_idx] is None:
                        local_idx += 1
                    self.frames[0] = self.frames[local_idx]
                else:
                    self.frames[frame_idx] = self.frames[frame_idx - 1]
                self.frames_queue.put((frame_idx, self.frames[frame_idx]))
            fqs = self.frames_queue.qsize()
            processed_number = self.landmarks_queue.qsize()
            bar.next(processed_number)
            while True:
                fqs = self.frames_queue.qsize()
                current_size = self.landmarks_queue.qsize()
                if current_size != processed_number:
                    bar.next(current_size - processed_number)
                    processed_number = current_size
                if current_size == current_batch_size:
                    break
            for _ in range(current_batch_size):
                frame_idx, vertices = self.landmarks_queue.get()
                if vertices is not None:
                    self.landmarks_list[frame_idx] = vertices
                    continue
                if frame_idx != 0:
                    self.landmarks_list[frame_idx] = self.landmarks_list[frame_idx - 1]
                else:
                    self.landmarks_list[frame_idx] = torch.Tensor(self.neutral_landmarks)[None,]
            landmarks_tensor = torch.cat(self.landmarks_list)
            if self.cuda:
                landmarks_tensor = landmarks_tensor.cuda()
            difference_tensor = landmarks_tensor - self.repeated_human_neutral_landmarks[:, :, :2]
            difference_tensor[MOUTH_LANDMARKS] *= 1.2
            left_eye_dirs, right_eye_dirs, nose_mouth_dirs = divide_landmarks_batch(difference_tensor)
            if self.cuda:
                left_eye_dirs = left_eye_dirs.cuda()
                right_eye_dirs = right_eye_dirs.cuda()
                nose_mouth_dirs = nose_mouth_dirs.cuda()
            self.execution_params.left_eye = self.repeated_model_neutral_landmarks[:, self.left_eye_indices][:, :, :2] + left_eye_dirs
            self.execution_params.right_eye = self.repeated_model_neutral_landmarks[:, self.right_eye_indices][:, :, :2] + right_eye_dirs
            self.execution_params.nose_mouth = self.repeated_model_neutral_landmarks[:, self.nose_mouth_indices][:, :, :2] + nose_mouth_dirs
            output = self.video_model.execute(self.execution_params)
            self.repeated_vertices[:, self.video_model.face_mask] = output[:, self.video_model.face_mask]
            yield current_batch_size, self.repeated_vertices.cpu(), self.frames

        bar.finish()
        yield None, None, None

    def process_frames_sequentially(self):
        frames_number = int(self.video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_num = frames_number // self.offline_mode_batch_size
        current_batch_size = self.offline_mode_batch_size
        bar = Bar('Video processing', max=frames_number, check_tty=False)

        for batch_idx in range(batch_num + 1):
            if batch_idx == batch_num:
                current_batch_size = frames_number - batch_num * self.offline_mode_batch_size
            for frame_idx in range(current_batch_size):
                bar.next()
                _, frame = self.video_stream.read()
                self.frames[frame_idx] = frame
                self.detector.get_image(frame)
                found, rect = self.detector.detect_face()
                if not found:
                    if frame_idx != 0:
                        self.landmarks_dirs[frame_idx] = self.landmarks_dirs[frame_idx - 1]
                    else:
                        self.landmarks_dirs[frame_idx] = torch.zeros_like(self.neutral_landmarks)
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
                self.landmarks_dirs[frame_idx] = torch.Tensor(smoothed_landmarks - self.neutral_landmarks)[None,]

            left_eye_dirs, right_eye_dirs, nose_mouth_dirs = divide_landmarks_batch(torch.cat(self.landmarks_dirs))
            if self.cuda:
                left_eye_dirs = left_eye_dirs.cuda()
                right_eye_dirs = right_eye_dirs.cuda()
                nose_mouth_dirs = nose_mouth_dirs.cuda()
            self.execution_params.left_eye = self.repeated_neutral_landmarks[:, self.left_eye_indices][:, :, :2] + left_eye_dirs
            self.execution_params.right_eye = self.repeated_neutral_landmarks[:, self.right_eye_indices][:, :, :2] + right_eye_dirs
            self.execution_params.nose_mouth = self.repeated_neutral_landmarks[:, self.nose_mouth_indices][:, :, :2] + nose_mouth_dirs
            output = self.video_model.execute(self.execution_params)
            self.repeated_vertices[:, self.video_model.face_mask] = output[:, self.video_model.face_mask]
            yield current_batch_size, self.repeated_vertices.cpu(), self.frames

        bar.finish()
        yield None, None, None

    def animate_mesh(self):
        vertices = torch.Tensor(self.video_model.neutral_vertices)
        if self.cuda:
            vertices = vertices.cuda()
        if self.video_mode == VideoMode.ONLINE:
            self.landmarks_number = 2
            while True:
                _, model_output = self.process_frame()
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
            #     input_frame, model_output = self.process_frame()
            #     if model_output is not None:
            #         vertices[self.video_model.face_mask, :] = model_output[0, self.video_model.face_mask]
            #     self.visualizer.render(vertices.cpu().numpy().squeeze(), input_frame)
            self.init_concurrent_mode()
            processed_frames = self.process_frames_concurrently()
            while True:
                current_batch_size, output_vertices, input_frames = next(processed_frames)
                if current_batch_size is None:
                    break
                output_vertices = output_vertices.numpy().squeeze()
                for idx in range(current_batch_size):
                    self.visualizer.render(output_vertices[idx], input_frames[idx])
            self.release_concurrent_mode()

    def stop(self):
        self.visualizer.release()
        self.video_stream.release()
        cv2.destroyAllWindows()
