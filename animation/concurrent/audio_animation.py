from audio_animation.audio_animation import AudioAnimation
import math


class AudioAnimationParams:
    def __init__(self,
                 cuda=True,
                 audio_path=None,
                 target_frame_rate=None,
                 target_frames_number=None,
                 origin_frame_rate=50):
        self.cuda = cuda
        self.audio_path = audio_path
        self.target_frame_rate = target_frame_rate
        self.target_frames_number = target_frames_number
        self.origin_frame_rate = origin_frame_rate


def audio_animation_pipeline(params: AudioAnimationParams, output_queue):
    animation = AudioAnimation(cuda=params.cuda, logging=False)
    animation.set_audio(audio_path=params.audio_path)
    processed_features = animation.audio_model.execute(animation.execution_params)
    need_adaptation = True
    if params.target_frame_rate == params.origin_frame_rate:
        need_adaptation = False
    supposed_origin_frames_number = params.origin_frame_rate * (params.target_frames_number / params.target_frame_rate)
    step_size = (supposed_origin_frames_number - 1) / (params.target_frames_number - 1)
    relative_current_position = 0
    left_item = None
    frame_idx = 0
    while True:
        current_batch_size, output_vertices = next(processed_features)
        if current_batch_size is None:
            output_queue.put(-1)
            break
        output_vertices = output_vertices.numpy().squeeze()
        if not need_adaptation:
            for idx in range(current_batch_size):
                output_queue.put((frame_idx, output_vertices[idx]))
                frame_idx += 1
            continue
        while relative_current_position < current_batch_size:
            left_idx = math.floor(relative_current_position)
            right_idx = left_idx + 1
            fractional_part = relative_current_position - left_idx
            if right_idx >= current_batch_size and fractional_part != 0:
                left_item = output_vertices[left_idx]
                break
            if left_idx >= 0:
                left_item = output_vertices[left_idx]
            if fractional_part == 0:
                output_queue.put((frame_idx, left_item))
            else:
                output_queue.put((frame_idx, (1 - fractional_part) * left_item + fractional_part * output_vertices[right_idx]))
            frame_idx += 1
            relative_current_position += step_size
        relative_current_position -= current_batch_size
