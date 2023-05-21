from video_animation.detector.detector import Detector
import torch
from utils.landmarks import align_landmarks


def transform_frame_to_landmarks(input_queue, output_queue):
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
        local_detector.visualize_bounding_box()
        landmarks = local_detector.detect_landmarks()
        local_detector.visualize_landmarks()
        output_queue.put((frame_idx, torch.Tensor(align_landmarks(landmarks))[None, ], local_detector.image))
