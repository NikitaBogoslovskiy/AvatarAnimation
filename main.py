from config.paths import PROJECT_DIR
from animation.animation import Animation
from video_animation.video_animation import VideoAnimation
from audio_animation.audio_animation import AudioAnimation


def video_animation_online():
    animation = VideoAnimation()
    animation.set_video()
    animation.capture_neutral_face()
    animation.animate_mesh()
    animation.stop()


def video_animation_offline(video_path, photo_path=None):
    animation = VideoAnimation()
    animation.set_video(video_path=video_path)
    if photo_path is None:
        animation.set_current_neutral_face()
    else:
        animation.capture_neutral_face(photo_path=photo_path)
    animation.animate_mesh()
    animation.stop()


def audio_animation_offline(audio_path):
    animation = AudioAnimation()
    animation.set_audio(audio_path=audio_path)
    animation.animate_mesh()


def overall_animation_offline(video_path, audio_support_level=1.0):
    animation = Animation(audio_support_level=audio_support_level)
    animation.set_parameters(video_path=video_path)
    animation.animate_mesh()


if __name__ == "__main__":
    # video_animation_online()
    # video_animation_offline(video_path=f"C:/Users/nikit/Pictures/Camera Roll/WIN_20230521_02_06_53_Pro.mp4")
    # audio_animation_offline(audio_path=f"{PROJECT_DIR}/other_data/input_audios/2.wav")
    overall_animation_offline(video_path=f"C:/Users/nikit/Pictures/Camera Roll/WIN_20230521_02_06_53_Pro.mp4", audio_support_level=1.0)
