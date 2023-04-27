import torch
import numpy as np
from video_animation.detector.detector import Detector

LEFT_EYEBROW_LANDMARKS = [17, 18, 19, 20, 21]
LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41]
RIGHT_EYEBROW_LANDMARKS = [22, 23, 24, 25, 26]
RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47]
NOSE_LANDMARKS = [27, 28, 29, 30, 31, 32, 33, 34, 35]
MOUTH_LANDMARKS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
JAW_LANDMARKS = [7, 8, 9]
FACIAL_LANDMARKS = [*LEFT_EYEBROW_LANDMARKS, *LEFT_EYE_LANDMARKS, *RIGHT_EYEBROW_LANDMARKS, *RIGHT_EYE_LANDMARKS,
                    *NOSE_LANDMARKS, *MOUTH_LANDMARKS, *JAW_LANDMARKS]


def divide_landmarks(landmarks: torch.Tensor):
    left_eye_indices = torch.tensor(np.array([*LEFT_EYEBROW_LANDMARKS, *LEFT_EYE_LANDMARKS]))
    right_eye_indices = torch.tensor(np.array([*RIGHT_EYEBROW_LANDMARKS, *RIGHT_EYE_LANDMARKS]))
    nose_mouth_indices = torch.tensor(np.array([*NOSE_LANDMARKS, *MOUTH_LANDMARKS, *JAW_LANDMARKS]))
    left_eye = landmarks[left_eye_indices, :]
    right_eye = landmarks[right_eye_indices, :]
    nose_mouth = landmarks[nose_mouth_indices, :]
    return left_eye, right_eye, nose_mouth


def divide_landmarks_batch(landmarks: torch.Tensor):
    left_eye_indices = torch.tensor(np.array([*LEFT_EYEBROW_LANDMARKS, *LEFT_EYE_LANDMARKS]))
    right_eye_indices = torch.tensor(np.array([*RIGHT_EYEBROW_LANDMARKS, *RIGHT_EYE_LANDMARKS]))
    nose_mouth_indices = torch.tensor(np.array([*NOSE_LANDMARKS, *MOUTH_LANDMARKS, *JAW_LANDMARKS]))
    left_eye = landmarks[:, left_eye_indices, :]
    right_eye = landmarks[:, right_eye_indices, :]
    nose_mouth = landmarks[:, nose_mouth_indices, :]
    return left_eye, right_eye, nose_mouth


def distance(x: np.array, y: np.array):
    return np.linalg.norm(x - y)


def align_landmarks(landmarks: np.array):
    new_landmarks = landmarks.astype("float")
    left_eye_center = np.mean(new_landmarks[LEFT_EYE_LANDMARKS], axis=0)
    right_eye_center = np.mean(new_landmarks[RIGHT_EYE_LANDMARKS], axis=0)
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                   (left_eye_center[1] + right_eye_center[1]) / 2)
    new_landmarks -= eyes_center
    right_eye_center -= eyes_center
    theta = np.arctan(right_eye_center[1] / right_eye_center[0])
    c, s = np.cos(theta), np.sin(theta)
    new_landmarks = np.dot(new_landmarks, np.array(((c, -s), (s, c))))
    shift = (new_landmarks[30][1] - new_landmarks[29][1]) / 2
    face_center = new_landmarks[27] + (0, 5 * shift)
    new_landmarks -= face_center
    # x_min = np.min(new_landmarks[:, 0])
    # y_min = np.min(new_landmarks[:, 1])
    width = np.max(new_landmarks[:, 0]) - np.min(new_landmarks[:, 0])
    new_landmarks /= width * 7
    new_landmarks *= (1, -1)
    # height = np.max(new_landmarks[:, 1]) - y_min
    # divider = width if width > height else height
    # new_landmarks = (new_landmarks - (x_min, y_min)) / (divider, divider)
    return new_landmarks


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


def transform_frame_to_landmarks(process_idx, input_queue, output_queue):
    local_detector = Detector()
    while True:
        if input_queue.empty():
            continue
        top = input_queue.get()
        if top == -1:
            return
        frame_idx, frame = top
        local_detector.get_image(frame)
        found, rect = local_detector.detect_face()
        if not found:
            output_queue.put((frame_idx, None))
            continue
        landmarks = local_detector.detect_landmarks()
        output_queue.put((frame_idx, torch.Tensor(align_landmarks(landmarks))[None, ]))
