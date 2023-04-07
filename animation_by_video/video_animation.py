import torch
from animation_by_video.detector.detector import Detector
import cv2
from animation_by_video.model.video_model import VideoModel, VideoModelExecuteParams
from animation_by_video.visualizer.visualizer import Visualizer
from collections import deque
import numpy as np
from utils.landmarks import align_landmarks, divide_landmarks, LEFT_EYE_LANDMARKS, LEFT_EYEBROW_LANDMARKS, \
    RIGHT_EYEBROW_LANDMARKS, RIGHT_EYE_LANDMARKS, MOUTH_LANDMARKS, NOSE_LANDMARKS, JAW_LANDMARKS, convert_lm_coordinates


class VideoAnimation:
    def __init__(self, index=0, cuda=True):
        self.detector = Detector()
        self.video_stream = cv2.VideoCapture(index)
        self.cuda = cuda
        self.video_model = VideoModel(self.cuda)
        self.video_model.load_model(weights_path="C:/Content/Python/AvatarAnimation/animation_by_video/weights/video_model_1_96900_eaUV2Qs70Tlmn0S.pt")
        self.video_model.init_for_execution(batch_size=1)
        self.visualizer = Visualizer()
        self.visualizer.set_surfaces(self.video_model.flame_model.flamelayer.faces)
        self.landmarks_number = 2
        self.landmarks_sum = np.zeros((68, 2))
        self.landmarks_history = deque()
        self.neutral_landmarks = None
        self.momentum = 0.0

    def capture_neutral_face(self):
        while True:
            ret, frame = self.video_stream.read()
            self.detector.get_image(frame)
            b, rect = self.detector.detect_face()
            if b:
                self.detector.visualize_bounding_box()
            cv2.imshow('Press "Enter" to capture neutral face', self.detector.image)
            if cv2.waitKey(1) == 13:
                self.detector.get_image(frame)
                b, rect = self.detector.detect_face()
                if b:
                    landmarks = self.detector.detect_landmarks()
                    self.neutral_landmarks = align_landmarks(landmarks)
                    cv2.destroyAllWindows()
                    break

    def animate_mesh(self):
        execution_params = VideoModelExecuteParams()
        left_eye_indices = torch.tensor(np.array([*LEFT_EYEBROW_LANDMARKS, *LEFT_EYE_LANDMARKS]))
        right_eye_indices = torch.tensor(np.array([*RIGHT_EYEBROW_LANDMARKS, *RIGHT_EYE_LANDMARKS]))
        nose_mouth_indices = torch.tensor(np.array([*NOSE_LANDMARKS, *MOUTH_LANDMARKS, *JAW_LANDMARKS]))
        vertices = torch.Tensor(self.video_model.neutral_vertices)
        if self.cuda:
            vertices = vertices.cuda()
        local_counter = 0
        while True:
            ret, frame = self.video_stream.read()
            self.detector.get_image(frame)
            b, rect = self.detector.detect_face()
            if b:
                landmarks = self.detector.detect_landmarks()
                self.detector.visualize_landmarks()
                landmarks = align_landmarks(landmarks)
                self.landmarks_history.append(landmarks)
                self.landmarks_sum += landmarks
                if local_counter == self.landmarks_number:
                    left_landmarks = self.landmarks_history.popleft()
                    self.landmarks_sum -= left_landmarks
                    smoothed_landmarks = self.landmarks_sum / self.landmarks_number
                else:
                    smoothed_landmarks = landmarks
                    local_counter += 1
                left_eye_dir, right_eye_dir, nose_mouth_dir = \
                    divide_landmarks(smoothed_landmarks - self.neutral_landmarks)
                left_eye_dir = torch.Tensor(left_eye_dir)
                right_eye_dir = torch.Tensor(right_eye_dir)
                nose_mouth_dir = torch.Tensor(nose_mouth_dir)
                if self.cuda:
                    left_eye_dir = left_eye_dir.cuda()
                    right_eye_dir = right_eye_dir.cuda()
                    nose_mouth_dir = nose_mouth_dir.cuda()
                execution_params.left_eye = self.video_model.neutral_landmarks[left_eye_indices][:, :2] + left_eye_dir
                execution_params.right_eye = self.video_model.neutral_landmarks[right_eye_indices][:, :2] + right_eye_dir
                execution_params.nose_mouth = self.video_model.neutral_landmarks[nose_mouth_indices][:, :2] + nose_mouth_dir
                vertices[self.video_model.face_mask, :] = self.video_model.execute(execution_params)[0, self.video_model.face_mask]
                self.visualizer.render(vertices.cpu().numpy().squeeze(), pause=0.001)
                cv2.imshow('Window for avatar manipulation', self.detector.image)
                if cv2.waitKey(1) == 27:
                    break

    def stop(self):
        self.visualizer.release()
        self.video_stream.release()
        cv2.destroyAllWindows()
