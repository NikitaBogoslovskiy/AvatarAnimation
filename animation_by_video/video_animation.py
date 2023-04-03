import torch
from matplotlib import pyplot as plt

from animation_by_video.detector.detector import Detector
import cv2
from animation_by_video.model.video_model import VideoModel, VideoModelExecuteParams
from animation_by_video.visualizer.visualizer import Visualizer
from queue import Queue
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
        self.lm_history = Queue()
        self.landmarks_direction = np.zeros((68, 2))
        self.lm_counter = 2
        self.lm_sum = np.zeros((68, 2))
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
                    # le = self.model.release_data['left_eye'][0]
                    # re = self.model.release_data['right_eye'][0]
                    # nm = self.model.release_data['nose_mouth'][0]
                    # puc = torch.cat([le, re, nm], axis=0)[:, :2].numpy()
                    # self.neutral_landmarks = new_lms[17:]
                    # lm.draw_it(new_rot_lms[17:], puc)
                    # print(new_lms)
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
        while True:
            ret, frame = self.video_stream.read()
            self.detector.get_image(frame)
            b, rect = self.detector.detect_face()
            if b:
                landmarks = self.detector.detect_landmarks()
                # self.det.visualize_box()
                self.detector.visualize_landmarks()
                landmarks = align_landmarks(landmarks)
                self.lm_history.put(landmarks)
                self.lm_sum += landmarks
                mean = landmarks
                if self.lm_counter == 0:
                    left = self.lm_history.get()
                    self.lm_sum -= left
                    mean = self.lm_sum / 2
                else:
                    self.lm_counter -= 1
                # self.landmarks_direction = self.momentum * self.landmarks_direction + landmarks - self.neutral_landmarks
                # mesh_n_lms = self.video_model.neutral_landmarks[:, :2].cpu().numpy().squeeze().T.tolist()
                # diff = mean - self.neutral_landmarks
                # mesh_lms = (self.video_model.neutral_landmarks[:, :2].cpu().numpy().squeeze() + diff).T.tolist()
                # if cv2.waitKey(1) == 13:
                #     plt.scatter(mesh_n_lms[0], mesh_n_lms[1], color="red")
                #     plt.scatter(mesh_lms[0], mesh_lms[1], color="green")
                #     plt.show()
                left_eye_dir, right_eye_dir, nose_mouth_dir = divide_landmarks(mean - self.neutral_landmarks)
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
                # diff = (mean - self.neutral_landmarks)
                # diff_l, diff_r, diff_nm = divide_landmarks(diff[None, :])
                # diff_l_t = torch.Tensor(diff_l)
                # diff_r_t = torch.Tensor(diff_r)
                # diff_nm_t = torch.Tensor(diff_nm)
                # le = self.model.release_data['left_eye'] + diff_l_t
                # re = self.model.release_data['right_eye'] + diff_r_t
                # nm = self.model.release_data['nose_mouth'] + diff_nm_t
                # output = self.video_model.execute(execution_params)
                vertices[self.video_model.face_mask, :] = self.video_model.execute(execution_params)[0, self.video_model.face_mask]
                self.visualizer.render(vertices.cpu().numpy().squeeze(), pause=0.001)
                cv2.imshow('Window for avatar manipulation', self.detector.image)
                if cv2.waitKey(1) == 27:
                    break

    def stop(self):
        self.visualizer.release()
        self.video_stream.release()
        cv2.destroyAllWindows()
