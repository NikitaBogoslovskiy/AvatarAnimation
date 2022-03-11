from imutils import face_utils
import imutils
import dlib
import cv2
import os
import pickle
import numpy as np
import plotly.graph_objects as go
import torch
import matplotlib.pyplot as plt


class Detector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('files/shape_predictor.dat')

    def get_image(self, image):
        self.cropped_image = imutils.resize(image, width=500)
        # self.cropped_image = image
        self.gray_image = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)
        self.corrected_image = self.gray_image

    def load_image(self, path):
        if not os.path.exists(path):
            raise IOError('Path is incorrect')
        self.original_image = cv2.imread(path)
        self.get_image(self.original_image)

    def detect_face(self):
        rects = self.detector(self.corrected_image, 1)
        if len(rects) == 0:
            return False, None
        else:
            self.rect = rects[0]
            return True, self.rect

    def detect_landmarks(self):
        shape = self.predictor(self.corrected_image, self.rect)
        self.landmarks = face_utils.shape_to_np(shape)[17:]
        return self.landmarks

    def visualize_box(self):
        self.new_image = self.cropped_image.copy()
        x, y, w, h = face_utils.rect_to_bb(self.rect)
        cv2.rectangle(self.new_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    def visualize_landmarks(self):
        for x, y in self.landmarks:
            cv2.circle(self.new_image, (x, y), 1, (0, 0, 255), -1)

    def show(self):
        cv2.imshow('Landmarks Detector: the result', self.new_image)
        cv2.waitKey(0)

    def save(self, path):
        cv2.imwrite(path, self.new_image)


def track_face(path=0):
    det = Detector()
    video = cv2.VideoCapture(path)
    while True:
        ret, frame = video.read()
        det.get_image(frame)
        if det.detect_face():
            det.visualize_box()
            det.detect_landmarks()
            det.visualize_landmarks()
            frame = det.new_image
        cv2.imshow('Video with box and landmarks', frame)
        if cv2.waitKey(1) != -1:
            break
    video.release()
    cv2.destroyAllWindows()


def load_mesh_landmarks(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    indices = np.array(data['lmk_face_idx'], dtype='int32')
    coordinates = data['lmk_b_coords']
    return indices, coordinates


def meshes_to_landmarks_numpy(vertices, surfaces, indices, coordinates):
    lm_faces = np.take_along_axis(surfaces, indices[:, None], 0).reshape(153, 1)
    lm_faces_coords = np.take_along_axis(vertices, lm_faces[None, :], 1).reshape(-1, 51, 3, 3)
    lm_coords = np.einsum('ij,lijk->lik', coordinates, lm_faces_coords)
    return lm_coords


def meshes_to_landmarks_torch(vertices, surfaces, indices, coordinates):
    lm_faces = torch.take_along_dim(surfaces, indices[:, None], dim=0).reshape(153, 1)
    lm_faces_coords = torch.take_along_dim(vertices, lm_faces[None, :], dim=1).reshape(-1, 51, 3, 3)
    lm_coords = torch.einsum('ij,lijk->lik', coordinates, lm_faces_coords)
    return lm_coords


def divide_landmarks(lm_coords):
    left_eye_lms = np.array([*range(0, 5), *range(19, 25)])
    right_eye_lms = np.array([*range(5, 10), *range(25, 31)])
    nose_mouth_lms = np.array([*range(10, 19), *range(31, 51)])
    left_eyes = lm_coords[:, left_eye_lms, :]
    right_eyes = lm_coords[:, right_eye_lms, :]
    noses_mouths = lm_coords[:, nose_mouth_lms, :]
    return left_eyes, right_eyes, noses_mouths


def draw(vertices, left_eye, right_eye, noses_mouth):
    vertices_color = np.zeros_like(vertices)
    left_eye_color = np.zeros_like(left_eye)
    right_eye_color = np.zeros_like(right_eye)
    nose_mouth_color = np.zeros_like(noses_mouth)
    vertices_color[:] = np.array([0, 220, 0])
    left_eye_color[:] = np.array([50, 0, 220])
    right_eye_color[:] = np.array([220, 0, 50])
    nose_mouth_color[:] = np.array([140, 0, 140])
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                mode='markers',
                marker=dict(size=1, color=vertices_color)
            ),
            go.Scatter3d(
                x=left_eye[:, 0], y=left_eye[:, 1], z=left_eye[:, 2],
                mode='markers',
                marker=dict(size=3, color=left_eye_color)
            ),
            go.Scatter3d(
                x=right_eye[:, 0], y=right_eye[:, 1], z=right_eye[:, 2],
                mode='markers',
                marker=dict(size=3, color=right_eye_color)
            ),
            go.Scatter3d(
                x=noses_mouth[:, 0], y=noses_mouth[:, 1], z=noses_mouth[:, 2],
                mode='markers',
                marker=dict(size=3, color=nose_mouth_color)
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True)
            )
        )
    )
    fig.show()


def draw_it(vertices, puc):
    plt.rcParams["figure.figsize"] = [8, 8]
    colors = np.resize(np.zeros_like(vertices), (51, 3))
    colors[:] = np.array([0, 220, 0]) / 256
    colors2 = np.resize(np.zeros_like(puc), (51, 3))
    colors2[:] = np.array([220, 0, 0]) / 256
    plt.scatter(vertices[:, 0], vertices[:, 1], c=colors)
    #plt.scatter(puc[:, 0], puc[:, 1], c=colors2)
    plt.show()
