from config.paths import PROJECT_DIR
import torch
from video_animation.detector.detector import Detector
import cv2
from video_animation.model.video_model import VideoModel, VideoModelExecuteParams
from video_animation.visualizer.offline_visualizer import OfflineVisualizer
from video_animation.visualizer.online_visualizer import OnlineVisualizer
import numpy as np
from utils.landmarks import align_landmarks, divide_landmarks, LEFT_EYE_LANDMARKS, LEFT_EYEBROW_LANDMARKS, \
    RIGHT_EYEBROW_LANDMARKS, RIGHT_EYE_LANDMARKS, MOUTH_LANDMARKS, NOSE_LANDMARKS, JAW_LANDMARKS, divide_landmarks_batch
from utils.video_settings import VideoMode, OfflineMode
import os
from multiprocessing import Process, Queue
from progress.bar import Bar
from collections import deque
from video_animation.concurrent.rendering import VisualizerParams, render_sequentially
from video_animation.concurrent.detection import transform_frame_to_landmarks


class VideoAnimation:
    def __init__(self, cuda=True, offline_mode_batch_size=100, show_detection_results=True, logging=True):
        self.video_stream = None
        self.video_mode = None
        self.online_visualizer = None
        self.offline_visualizer = None
        self.visualizer_params = None
        self.neutral_landmarks = None
        self.offline_mode = OfflineMode.CONCURRENT
        self.offline_mode_batch_size = offline_mode_batch_size
        self.detector = Detector()
        self.cuda = cuda
        self.show_detection_results = show_detection_results
        self.logging = logging
        self.video_model = VideoModel(self.cuda)
        self.video_model.load_model(
            weights_path=f"{PROJECT_DIR}/video_animation/weights/video_model_1_98800_12.06.2023-03.05.35.pt")
        self.landmarks_sum = np.zeros((68, 2))
        self.landmarks_history = deque()
        self.local_counter = 0
        self.execution_params = VideoModelExecuteParams()
        self.left_eye_indices = torch.tensor(np.array([*LEFT_EYEBROW_LANDMARKS, *LEFT_EYE_LANDMARKS]))
        self.right_eye_indices = torch.tensor(np.array([*RIGHT_EYEBROW_LANDMARKS, *RIGHT_EYE_LANDMARKS]))
        self.nose_mouth_indices = torch.tensor(np.array([*NOSE_LANDMARKS, *MOUTH_LANDMARKS, *JAW_LANDMARKS]))
        self.pre_weights = None
        self.post_weights = None
        # self.smoothing_level = 5
        # self.weights = [0.05, 0.05, 0.1, 0.2, 0.6]
        self.smoothing_level = 2
        self.weights = [0.3, 0.7]
        self.last_landmarks = deque()
        self.last_vertices = deque()
        self.repeated_human_neutral_landmarks = None
        self.landmarks_list = None
        self.processes_number = None
        self.repeated_model_neutral_landmarks = None
        self.repeated_vertices = None
        self.landmarks_queue = None
        self.frames_queue = None
        self.frames = None
        self.processes = None
        self.ready_to_render_queue = None
        self.rendering = None
        self.repeated_neutral_landmarks = None
        self.landmarks_dirs = None

    def set_video(self, video_path=0):
        if video_path == 0:
            self.video_mode = VideoMode.ONLINE
            self.online_visualizer = OnlineVisualizer()
            self.video_stream = cv2.VideoCapture(video_path)
            self.video_model.init_for_execution(batch_size=1)
            self.online_visualizer.set_surfaces(self.video_model.flame_model.flamelayer.faces)
        else:
            self.video_mode = VideoMode.OFFLINE
            if not os.path.isfile(video_path):
                raise FileNotFoundError("File path is incorrect")
            self.video_stream = cv2.VideoCapture(video_path)
            directory = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.video_model.init_for_execution(batch_size=self.offline_mode_batch_size)
            self.visualizer_params = VisualizerParams(save_path=f"{directory}/{video_name}_output.mp4",
                                                      surfaces=self.video_model.flame_model.flamelayer.faces,
                                                      width=int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                      height=int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                                      frame_rate=int(self.video_stream.get(cv2.CAP_PROP_FPS)))

    def init_settings(self):
        self.video_mode = VideoMode.OFFLINE
        self.video_model.init_for_execution(batch_size=self.offline_mode_batch_size)

    def init_concurrent_mode(self, processes_number=8):
        if self.logging:
            print("Initializing concurrent mode... ", end='')
        self.repeated_human_neutral_landmarks = torch.Tensor(self.neutral_landmarks)[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        if self.cuda:
            self.repeated_human_neutral_landmarks = self.repeated_human_neutral_landmarks.cuda()
        self.landmarks_list = [torch.Tensor(self.neutral_landmarks)[None,]] * self.offline_mode_batch_size
        self.processes_number = processes_number
        self.repeated_model_neutral_landmarks = self.video_model.neutral_landmarks[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        self.repeated_vertices = torch.Tensor(self.video_model.neutral_vertices)[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        self.landmarks_queue = Queue()
        self.frames_queue = Queue()
        self.frames = [None] * self.offline_mode_batch_size
        self.processes = []
        for process_idx in range(self.processes_number):
            self.processes.append(Process(target=transform_frame_to_landmarks, args=(self.frames_queue, self.landmarks_queue)))
            self.processes[process_idx].start()
        if self.logging:
            print("Done.")

    def release_concurrent_mode(self):
        for process_idx in range(self.processes_number):
            self.frames_queue.put(-1)
        for process_idx in range(self.processes_number):
            self.processes[process_idx].join()

    def init_concurrent_rendering(self):
        if self.logging:
            print("Initializing concurrent rendering... ", end='')
        self.ready_to_render_queue = Queue()
        self.rendering = Process(target=render_sequentially, args=(self.visualizer_params, self.ready_to_render_queue,))
        self.rendering.start()
        if self.logging:
            print("Done.")

    def release_concurrent_rendering(self):
        self.ready_to_render_queue.put(-1)
        self.rendering.join()

    def init_sequential_mode(self):
        if self.logging:
            print("Initializing sequential mode... ", end='')
        self.repeated_human_neutral_landmarks = torch.Tensor(self.neutral_landmarks)[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        if self.cuda:
            self.repeated_human_neutral_landmarks = self.repeated_human_neutral_landmarks.cuda()
        self.repeated_model_neutral_landmarks = self.video_model.neutral_landmarks[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        self.repeated_vertices = torch.Tensor(self.video_model.neutral_vertices)[None, :, :].repeat(self.offline_mode_batch_size, 1, 1)
        # if self.cuda:
        #     self.repeated_vertices = self.repeated_vertices.cuda()
        self.landmarks_list = [torch.zeros_like(torch.Tensor(self.neutral_landmarks)[None,])] * self.offline_mode_batch_size
        self.frames = [None] * self.offline_mode_batch_size
        if self.logging:
            print("Done.")

    def release_sequential_mode(self):
        pass

    def init_sequential_rendering(self):
        if self.logging:
            print("Initializing sequential rendering... ", end='')
        self.offline_visualizer = OfflineVisualizer(self.visualizer_params.save_path)
        self.offline_visualizer.set_surfaces(self.visualizer_params.surfaces)
        self.offline_visualizer.init_settings(animation_resolution=(self.visualizer_params.height, self.visualizer_params.height),
                                              input_resolution=(self.visualizer_params.width, self.visualizer_params.height),
                                              frame_rate=self.visualizer_params.frame_rate)
        if self.logging:
            print("Done.")

    def release_sequential_rendering(self):
        self.offline_visualizer.release()

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

    def _set_pre_weights_offline(self):
        pre_weights_list = []
        pre_template_tensor = torch.ones_like(torch.Tensor(self.neutral_landmarks))
        if self.cuda:
            pre_template_tensor = pre_template_tensor.cuda()
        for w in self.weights:
            pre_weights_list.append((pre_template_tensor * w)[None, :])
        self.pre_weights = torch.cat(pre_weights_list)

    def _set_post_weights_offline(self):
        post_weights_list = []
        post_template_tensor = torch.ones_like(self.video_model.neutral_vertices)
        for w in self.weights:
            post_weights_list.append((post_template_tensor * w)[None, :])
        self.post_weights = torch.cat(post_weights_list)

    def _set_pre_weights_online(self):
        pre_weights_list = []
        pre_template_tensor = np.ones_like(self.neutral_landmarks)
        for w in self.weights:
            weighted_tensor = pre_template_tensor * w
            pre_weights_list.append(weighted_tensor[None, :])
        self.pre_weights = np.concatenate(pre_weights_list)

    def _set_post_weights_online(self):
        self._set_post_weights_offline()

    def capture_neutral_face(self, photo_path=None):
        if self.video_mode == VideoMode.ONLINE:
            enter_pressed = False
            while True:
                ret, frame = self.video_stream.read()
                self.detector.get_image(frame)
                box_found, rect = self.detector.detect_face()
                if box_found:
                    self.detector.visualize_bounding_box()
                cv2.imshow('Press "Enter" to capture neutral face', self.detector.image)
                if cv2.waitKey(1) == 13:
                    enter_pressed = True
                if enter_pressed and box_found:
                    landmarks = self.detector.detect_landmarks()
                    if landmarks.shape[0] == 0:
                        continue
                    self.neutral_landmarks = align_landmarks(landmarks)
                    break
            cv2.destroyAllWindows()
        else:
            neutral_face_image = self._capture_neutral_face_from_photo(photo_path)
            self.detector.get_image(neutral_face_image)
            box_found, rect = self.detector.detect_face()
            if box_found:
                landmarks = self.detector.detect_landmarks()
                if landmarks.shape[0] == 0:
                    print("Error: cannot detect facial landmarks on photo")
                    exit(-1)
                self.neutral_landmarks = align_landmarks(landmarks)
            else:
                print("Error: cannot find face on photo")
                exit(-1)

    def _capture_neutral_face_from_video(self):
        enter_pressed = False
        while True:
            ret, frame = self.video_stream.read()
            self.detector.get_image(frame)
            found, rect = self.detector.detect_face()
            if found:
                self.detector.visualize_bounding_box()
            cv2.imshow('Press "Enter" to capture neutral face', self.detector.image)
            if cv2.waitKey(1) == 13:
                enter_pressed = True
            if enter_pressed and found:
                return frame

    @staticmethod
    def _capture_neutral_face_from_photo(photo_path):
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
            landmarks = align_landmarks(landmarks)
            if self.show_detection_results:
                self.detector.visualize_bounding_box()
                self.detector.visualize_landmarks()
            if len(self.last_landmarks) < self.smoothing_level:
                self.last_landmarks.append(landmarks[None, :])
                smoothed_landmarks = landmarks
            else:
                self.last_landmarks.popleft()
                self.last_landmarks.append(landmarks[None, :])
                smoothed_landmarks = np.sum(np.concatenate(list(self.last_landmarks)) * self.pre_weights, axis=0)
                self.last_landmarks[-1] = smoothed_landmarks[None, :]
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
            output = self.video_model.execute(self.execution_params)
            if len(self.last_vertices) < self.smoothing_level:
                self.last_vertices.append(output)
            else:
                self.last_vertices.popleft()
                self.last_vertices.append(output)
                output = torch.sum(torch.cat(list(self.last_vertices)) * self.post_weights, dim=0)[None, :]
                self.last_vertices[-1] = output
            return frame, output
        return frame, None

    def process_frames_concurrently(self):
        frames_number = int(self.video_stream.get(cv2.CAP_PROP_FRAME_COUNT)) - int(self.video_stream.get(cv2.CAP_PROP_POS_FRAMES))
        batch_num = frames_number // self.offline_mode_batch_size
        current_batch_size = self.offline_mode_batch_size
        if self.logging:
            bar = Bar('Video processing', max=frames_number, check_tty=False)
        iterations_number = batch_num + 1 if frames_number % self.offline_mode_batch_size != 0 else batch_num
        self._set_pre_weights_offline()
        self._set_post_weights_offline()

        for batch_idx in range(iterations_number):
            lacking_frames = []
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
            processed_number = self.landmarks_queue.qsize()
            if self.logging:
                bar.next(processed_number)
            while True:
                current_size = self.landmarks_queue.qsize()
                if current_size != processed_number:
                    if self.logging:
                        bar.next(current_size - processed_number)
                    processed_number = current_size
                if current_size == current_batch_size:
                    break
            for _ in range(current_batch_size):
                frame_idx, vertices, image = self.landmarks_queue.get()
                if self.show_detection_results:
                    self.frames[frame_idx] = image
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
            for landmarks_idx in range(current_batch_size):
                if len(self.last_landmarks) < self.smoothing_level:
                    self.last_landmarks.append(landmarks_tensor[landmarks_idx][None, :])
                else:
                    self.last_landmarks.popleft()
                    self.last_landmarks.append(landmarks_tensor[landmarks_idx][None, :])
                    landmarks_tensor[landmarks_idx] = torch.sum(torch.cat(list(self.last_landmarks)) * self.pre_weights, dim=0)
                    self.last_landmarks[-1] = landmarks_tensor[landmarks_idx][None, :]
            difference_tensor = landmarks_tensor - self.repeated_human_neutral_landmarks[:, :, :2]
            # difference_tensor[:, MOUTH_LANDMARKS] *= 1.0
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
            for vertices_idx in range(current_batch_size):
                if len(self.last_vertices) < self.smoothing_level:
                    self.last_vertices.append(self.repeated_vertices[vertices_idx][None, :])
                else:
                    self.last_vertices.popleft()
                    self.last_vertices.append(self.repeated_vertices[vertices_idx][None, :])
                    self.repeated_vertices[vertices_idx] = torch.sum(torch.cat(list(self.last_vertices)) * self.post_weights, dim=0)
                    self.last_vertices[-1] = self.repeated_vertices[vertices_idx][None, :]
            yield current_batch_size, self.repeated_vertices.cpu(), self.frames

        if self.logging:
            bar.finish()
        yield None, None, None

    def process_frames_sequentially(self):
        frames_number = int(self.video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_num = frames_number // self.offline_mode_batch_size
        current_batch_size = self.offline_mode_batch_size
        bar = Bar('Video processing', max=frames_number, check_tty=False)
        self._set_pre_weights_offline()
        self._set_post_weights_offline()

        for batch_idx in range(batch_num + 1):
            lacking_frames = []
            if batch_idx == batch_num:
                current_batch_size = frames_number - batch_num * self.offline_mode_batch_size
            for frame_idx in range(current_batch_size):
                success, frame = self.video_stream.read()
                if not success:
                    lacking_frames.append(frame_idx)
                    self.frames[frame_idx] = None
                else:
                    self.frames[frame_idx] = frame
            for frame_idx in lacking_frames:
                if frame_idx == 0:
                    local_idx = 1
                    while self.frames[local_idx] is None:
                        local_idx += 1
                    self.frames[0] = self.frames[local_idx]
                else:
                    self.frames[frame_idx] = self.frames[frame_idx - 1]
            for frame_idx in range(current_batch_size):
                self.detector.get_image(self.frames[frame_idx])
                found, rect = self.detector.detect_face()
                if not found:
                    if frame_idx != 0:
                        self.landmarks_list[frame_idx] = self.landmarks_list[frame_idx - 1]
                    else:
                        self.landmarks_list[frame_idx] = torch.zeros_like(self.neutral_landmarks)
                else:
                    landmarks = self.detector.detect_landmarks()
                    self.landmarks_list[frame_idx] = torch.Tensor(align_landmarks(landmarks))[None, ]
                    if self.show_detection_results:
                        self.detector.visualize_bounding_box()
                        self.detector.visualize_landmarks()
                        self.frames[frame_idx] = self.detector.image
                bar.next()
            landmarks_tensor = torch.cat(self.landmarks_list)
            if self.cuda:
                landmarks_tensor = landmarks_tensor.cuda()
            for landmarks_idx in range(current_batch_size):
                if len(self.last_landmarks) < self.smoothing_level:
                    self.last_landmarks.append(landmarks_tensor[landmarks_idx][None, :])
                else:
                    self.last_landmarks.popleft()
                    self.last_landmarks.append(landmarks_tensor[landmarks_idx][None, :])
                    landmarks_tensor[landmarks_idx] = torch.sum(torch.cat(list(self.last_landmarks)) * self.pre_weights, dim=0)
                    self.last_landmarks[-1] = landmarks_tensor[landmarks_idx][None, :]
            difference_tensor = landmarks_tensor - self.repeated_human_neutral_landmarks[:, :, :2]
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
            for vertices_idx in range(current_batch_size):
                if len(self.last_vertices) < self.smoothing_level:
                    self.last_vertices.append(self.repeated_vertices[vertices_idx][None, :])
                else:
                    self.last_vertices.popleft()
                    self.last_vertices.append(self.repeated_vertices[vertices_idx][None, :])
                    self.repeated_vertices[vertices_idx] = torch.sum(torch.cat(list(self.last_vertices)) * self.post_weights, dim=0)
                    self.last_vertices[-1] = self.repeated_vertices[vertices_idx][None, :]
            yield current_batch_size, self.repeated_vertices.cpu(), self.frames
        bar.finish()
        yield None, None, None

    def animate_mesh(self):
        if self.video_mode == VideoMode.ONLINE:
            self._set_pre_weights_online()
            self._set_post_weights_online()
            vertices = torch.Tensor(self.video_model.neutral_vertices)
            if self.cuda:
                vertices = vertices.cuda()
            while True:
                _, model_output = self.process_frame()
                if model_output is not None:
                    vertices[self.video_model.face_mask, :] = model_output[0, self.video_model.face_mask]
                self.online_visualizer.render(vertices.cpu().numpy().squeeze(), pause=0.001)
                cv2.imshow('Window for avatar manipulation', self.detector.image)
                if cv2.waitKey(1) == 27:
                    break
        elif self.video_mode == VideoMode.OFFLINE:
            if self.offline_mode == OfflineMode.CONCURRENT:
                self.init_concurrent_mode(processes_number=7)
                self.init_concurrent_rendering()
                processed_frames = self.process_frames_concurrently()
                while True:
                    current_batch_size, output_vertices, input_frames = next(processed_frames)
                    if current_batch_size is None:
                        break
                    output_vertices = output_vertices.numpy().squeeze()
                    for idx in range(current_batch_size):
                        self.ready_to_render_queue.put((output_vertices[idx], input_frames[idx]))
                if self.logging:
                    print("Rendering... ", end='')
                while self.ready_to_render_queue.qsize() != 0:
                    pass
                if self.logging:
                    print("Done.")
                self.release_concurrent_mode()
                self.release_concurrent_rendering()
            elif self.offline_mode == OfflineMode.BATCH:
                self.init_sequential_mode()
                self.init_sequential_rendering()
                processed_frames = self.process_frames_sequentially()
                while True:
                    current_batch_size, output_vertices, input_frames = next(processed_frames)
                    if current_batch_size is None:
                        break
                    output_vertices = output_vertices.numpy().squeeze()
                    for idx in range(current_batch_size):
                        self.offline_visualizer.render(output_vertices[idx], input_frames[idx])
                if self.logging:
                    print("Done.")
                self.release_sequential_mode()
                self.release_sequential_rendering()

    def stop(self):
        if self.logging and self.video_mode == VideoMode.OFFLINE:
            print(f"Output video has been saved to '{self.visualizer_params.save_path}'")
        if self.online_visualizer is not None:
            self.online_visualizer.release()
        self.video_stream.release()
        cv2.destroyAllWindows()
