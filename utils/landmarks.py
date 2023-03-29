import torch
import numpy as np
import cv2

LEFT_EYEBROW_LANDMARKS = [17, 18, 19, 20, 21]
LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41]
RIGHT_EYEBROW_LANDMARKS = [22, 23, 24, 25, 26]
RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47]
NOSE_LANDMARKS = [27, 28, 29, 30, 31, 32, 33, 34, 35]
MOUTH_LANDMARKS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]


def divide_landmarks(landmarks: torch.Tensor):
    left_eye_indices = torch.tensor(np.array([*LEFT_EYEBROW_LANDMARKS, *LEFT_EYE_LANDMARKS]))
    right_eye_indices = torch.tensor(np.array([*RIGHT_EYEBROW_LANDMARKS, *RIGHT_EYE_LANDMARKS]))
    nose_mouth_indices = torch.tensor(np.array([*NOSE_LANDMARKS, *MOUTH_LANDMARKS]))
    left_eye = landmarks[left_eye_indices, :]
    right_eye = landmarks[right_eye_indices, :]
    nose_mouth = landmarks[nose_mouth_indices, :]
    return left_eye, right_eye, nose_mouth

def distance(x: np.array, y: np.array):
    return np.linalg.norm(x - y)


def align_landmarks(landmarks: np.array):
    new_landmarks = landmarks.astype("float")
    left_eye_center = np.mean(landmarks[LEFT_EYE_LANDMARKS], axis=0)
    right_eye_center = np.mean(landmarks[RIGHT_EYE_LANDMARKS], axis=0)
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                   (left_eye_center[1] + right_eye_center[1]) / 2)
    new_landmarks -= eyes_center
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    theta = np.arctan2(dy, dx)
    c, s = np.cos(theta), np.sin(theta)
    new_landmarks = np.dot(new_landmarks, np.array(((c, -s), (s, c))))
    x_min = np.min(new_landmarks[:, 0])
    y_min = np.min(new_landmarks[:, 1])
    width = np.max(new_landmarks[:, 0]) - x_min
    height = np.max(new_landmarks[:, 1]) - y_min
    divider = width if width > height else height
    new_landmarks = (new_landmarks - (x_min, y_min)) / (divider, divider)
    return new_landmarks
