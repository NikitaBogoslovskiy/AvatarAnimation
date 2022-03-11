import torch
import face.landmarks as lm
import cv2
from imutils import face_utils
import numpy as np
from model.wrapper import ModelWrapper
from mesh.visualizer import Visualizer
import queue
import keyboard
import matplotlib.pyplot as plt


def convert_lm_coordinates(lms):
    lms[:, 1] *= -1
    lms = lms.astype('float32')
    middle = (lms[2] + lms[14]) / 2
    lms[:] -= middle
    right = np.array([lms[14, 0], 0])
    unit_vector1 = right / np.linalg.norm(right)
    unit_vector2 = lms[14] / np.linalg.norm(lms[14])
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product)
    cosa = np.cos(angle)
    sina = np.sin(angle)
    z = 0
    if lms[14, 1] > 0:
        z = 1
    else:
        z = -1
    rotate_matrix = np.array([[cosa, z * sina], [-z * sina, cosa]])
    rotated_lms = np.dot(rotate_matrix, lms.transpose()).transpose()
    width = rotated_lms[14, 0] - rotated_lms[2, 0]
    scaled_rotated_lms = rotated_lms / (width * 8)
    scaled_rotated_lms[:, 1] -= 0.008
    return scaled_rotated_lms


class Animation:
    def __init__(self, index=0):
        self.det = lm.Detector()
        self.video = cv2.VideoCapture(index)
        self.model = ModelWrapper()
        self.model.load_model()
        self.model.load_release_data()
        self.vis = Visualizer()
        self.vis.set_surfaces(self.model.release_data['surfaces'])
        self.lm_history = queue.Queue()
        self.lm_sum = np.zeros((51, 2))
        self.lm_counter = 2

    def capture_neutral_face(self):
        while True:
            ret, frame = self.video.read()
            self.det.get_image(frame)
            b, rect = self.det.detect_face()
            if b:
                self.det.visualize_box()
            cv2.imshow('Press "Enter" to capture neutral face', self.det.new_image)
            if cv2.waitKey(1) == 13:
                self.det.get_image(frame)
                b, rect = self.det.detect_face()
                if b:
                    lms = self.det.detect_landmarks()
                    new_lms = convert_lm_coordinates(lms)
                    # le = self.model.release_data['left_eye'][0]
                    # re = self.model.release_data['right_eye'][0]
                    # nm = self.model.release_data['nose_mouth'][0]
                    # puc = torch.cat([le, re, nm], axis=0)[:, :2].numpy()
                    self.neutral_landmarks = new_lms[17:]
                    # lm.draw_it(new_rot_lms[17:], puc)
                    # print(new_lms)
                    cv2.destroyAllWindows()
                    break

    def animate_mesh(self):
        while True:
            ret, frame = self.video.read()
            self.det.get_image(frame)
            b, rect = self.det.detect_face()
            if b:
                lms = self.det.detect_landmarks()
                # self.det.visualize_box()
                self.det.visualize_landmarks()
                new_lms = convert_lm_coordinates(lms)[17:]
                self.lm_history.put(new_lms)
                self.lm_sum += new_lms
                mean = new_lms
                if self.lm_counter == 0:
                    left = self.lm_history.get()
                    self.lm_sum -= left
                    mean = self.lm_sum / 2
                else:
                    self.lm_counter -= 1
                diff = (mean - self.neutral_landmarks)
                diff_l, diff_r, diff_nm = lm.divide_landmarks(diff[None, :])
                diff_l_t = torch.Tensor(diff_l)
                diff_r_t = torch.Tensor(diff_r)
                diff_nm_t = torch.Tensor(diff_nm)
                le = self.model.release_data['left_eye'] + diff_l_t
                re = self.model.release_data['right_eye'] + diff_r_t
                nm = self.model.release_data['nose_mouth'] + diff_nm_t
                vertices = self.model.execute(le, re, nm)
                self.vis.render(vertices[0], pause=0.001)
                cv2.imshow('Window for avatar manipulation', self.det.new_image)
                if cv2.waitKey(1) == 27:
                    break

    def stop(self):
        self.vis.release()
        self.video.release()
        cv2.destroyAllWindows()


class Counter:
    def __init__(self, init):
        self.value = init
        self.possible = True

    def inc(self):
        if self.possible:
            self.value += 1

    def turn_off(self):
        self.possible = False
