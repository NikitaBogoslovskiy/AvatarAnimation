from video_animation.video_animation import VideoAnimation


class VideoAnimationParams:
    def __init__(self,
                 cuda=True,
                 video_path=None,
                 neutral_face_path=None):
        self.cuda = cuda
        self.video_path = video_path
        self.neutral_face_path = neutral_face_path


def video_animation_pipeline(params: VideoAnimationParams, output_queue):
    animation = VideoAnimation(cuda=params.cuda)
    animation.set_video(video_path=params.video_path)
    if params.neutral_face_path is None:
        animation.set_current_neutral_face()
    else:
        animation.capture_neutral_face(photo_path=params.neutral_face_path)
    animation.init_concurrent_mode(processes_number=7)
    processed_frames = animation.process_frames_concurrently()
    frame_idx = 0
    while True:
        current_batch_size, output_vertices, input_frames = next(processed_frames)
        if current_batch_size is None:
            output_queue.put(-1)
            break
        output_vertices = output_vertices.numpy().squeeze()
        for idx in range(current_batch_size):
            output_queue.put((frame_idx, output_vertices[idx], input_frames[idx]))
            frame_idx += 1
    animation.release_concurrent_mode()
