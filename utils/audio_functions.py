import torch


def adapt_audio_to_frame_rate(input_audio_features, target_frame_rate, origin_frame_rate=50):
    if target_frame_rate == origin_frame_rate:
        return input_audio_features
    features_length = len(input_audio_features)
    step_size = origin_frame_rate / target_frame_rate
    current_position = step_size / 2
    output_audio_features = []
    while current_position <= features_length - 1:
        left_idx = int(current_position)
        right_idx = left_idx + 1
        fractional_part = current_position - left_idx
        if fractional_part == 0:
            output_audio_features.append(input_audio_features[left_idx])
        else:
            output_audio_features.append(((1 - fractional_part) * input_audio_features[left_idx] + fractional_part * input_audio_features[right_idx])[None, :])
        current_position += step_size
    return torch.cat(output_audio_features)
