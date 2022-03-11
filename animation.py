import torch
import face.landmarks as lm
import cv2
from imutils import face_utils
import numpy as np
from model.wrapper import ModelWrapper
from mesh.visualizer import Visualizer
import queue


def rotate_lm_coordinates(lms):
    lms -= lms[13]
    p1 = lms[10]
    p2 = lms[13]
    p3 = np.array([0, p1[1]])
    catet1 = np.sqrt(np.square(p1 - p3).sum())
    catet2 = np.sqrt(np.square(p2 - p3).sum())
    hypo = np.sqrt(catet1 ** 2 + catet2 ** 2)
    sina = catet1 / hypo
    cosa = catet2 / hypo
    s = 0
    if p1[0] > p2[0]:
        s = 1
    else:
        s = -1
    mat = np.array([[cosa, s * sina], [-s * sina, cosa]])
    return np.dot(lms, mat)


def convert_lm_coordinates(rect, lms):
    rect_x, rect_y, w, h = face_utils.rect_to_bb(rect)
    new_lms = np.zeros((len(lms), 2))
    for i in range(len(lms)):
        lm_x, lm_y = lms[i]
        lm_x -= rect_x
        lm_y -= rect_y
        new_lms[i, 0] = (lm_x - w/2) / (w * 7)
        new_lms[i, 1] = (-lm_y + h/2) / (h * 7)
    return new_lms


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
        self.lm_counter = Counter(0)

    def capture_neutral_face(self):
        while True:
            ret, frame = self.video.read()
            cv2.imshow('Output window', frame)
            if cv2.waitKey(1) == 13:
                self.det.get_image(frame)
                b, rect = self.det.detect_face()
                if b:
                    lms = self.det.detect_landmarks()
                    new_lms = convert_lm_coordinates(rect, lms)
                    new_rot_lms = rotate_lm_coordinates(new_lms)
                    # le = self.model.release_data['left_eye'][0]
                    # re = self.model.release_data['right_eye'][0]
                    # nm = self.model.release_data['nose_mouth'][0]
                    # puc = torch.cat([le, re, nm], axis=0)[:, :2].numpy()
                    self.neutral_landmarks = new_rot_lms
                    # lm.draw_it(new_rot_lms, puc)
                    # print(new_lms)
                    break

    def animate_mesh(self):
        while True:
            ret, frame = self.video.read()
            self.det.get_image(frame)
            b, rect = self.det.detect_face()
            if b:
                lms = self.det.detect_landmarks()
                self.det.visualize_box()
                self.det.visualize_landmarks()
                new_lms = convert_lm_coordinates(rect, lms)
                new_rot_lms = rotate_lm_coordinates(new_lms)
                self.lm_history.put(new_rot_lms)
                self.lm_sum += new_rot_lms
                self.lm_counter.inc()
                mean = new_rot_lms
                if self.lm_counter.value > 3:
                    self.lm_counter.turn_off()
                    left = self.lm_history.get()
                    self.lm_sum -= left
                    mean = self.lm_sum / 3
                diff = (mean - self.neutral_landmarks)
                diff_l, diff_r, diff_nm = lm.divide_landmarks(diff[None, :])
                diff_l_t = torch.Tensor(diff_l)
                diff_r_t = torch.Tensor(diff_r)
                diff_nm_t = torch.Tensor(diff_nm)
                le = self.model.release_data['left_eye'] + diff_l_t
                re = self.model.release_data['right_eye'] + diff_r_t
                nm = self.model.release_data['nose_mouth'] + diff_nm_t
                vertices = self.model.execute(le, re, nm)
                cv2.imshow('Output window', self.det.new_image)
                self.vis.render(vertices[0], pause=0.000001)
                if cv2.waitKey(1) == 27:
                    break

    def stop(self):
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
