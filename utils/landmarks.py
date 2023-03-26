import torch
import numpy as np


def divide_landmarks(landmarks):
    left_eye_indices = torch.tensor(np.array([*range(17, 22), *range(36, 42)]))
    right_eye_indices = torch.tensor(np.array([*range(22, 27), *range(42, 48)]))
    nose_mouth_indices = torch.tensor(np.array([*range(27, 36), *range(48, 68)]))
    left_eye = landmarks[left_eye_indices, :]
    right_eye = landmarks[right_eye_indices, :]
    nose_mouth = landmarks[nose_mouth_indices, :]
    return left_eye, right_eye, nose_mouth
