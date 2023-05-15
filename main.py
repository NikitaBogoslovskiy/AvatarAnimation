from config.paths import PROJECT_DIR
from animation.animation import Animation
from video_animation.video_animation import VideoAnimation


# tracker = FaceTracker()
# tracker.execute()
# vis = Visualizer()
# fm = FlameModel(get_config())
# shape_params = torch.zeros(1, 100).cuda()  # 35, 45, 90, 0..30, -4..5, -6..8
# pose_params_numpy = np.array([[0, 0, 0, 25 * RADIAN, 0 * RADIAN, 0 * RADIAN]], dtype=np.float32)
# pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()
# expression_params = torch.zeros(1, 50, dtype=torch.float32).cuda()
# v, l = fm.generate(shape_params, pose_params, expression_params)
# vis.set_surfaces(fm.flamelayer.faces)
# vis.render(v.detach().cpu().numpy().squeeze())
# v = VideoAnimation()
# v.set_video("C:/Users/nikit/Pictures/Camera Roll/WIN_20230421_22_29_43_Pro.mp4")
# v.capture_neutral_face("C:/Users/nikit/Pictures/Camera Roll/WIN_20230421_22_10_30_Pro.jpg")
# v.process_frames_2()
# v.animate_mesh()
# v = VideoAnimation()
# v.set_video()
# v.capture_neutral_face()
# v.animate_mesh()
# Dataset.generate(video_folder="C:/Content/Python/AvatarAnimation/audio_animation/dataset/raw_data",
#                  save_folder="C:/Content/Python/AvatarAnimation/audio_animation/dataset/train_data")

if __name__ == "__main__":
    # v = VideoAnimation()
    # # v.set_video(f"C:/Users/nikit/Pictures/Camera Roll/WIN_20230427_17_20_14_Pro.mp4")
    # # v.capture_neutral_face("C:/Users/nikit/Pictures/Camera Roll/WIN_20230427_17_19_40_Pro.jpg")
    # v.set_video(f"{PROJECT_DIR}/audio_animation/dataset/raw_data/GH035257_52.mp4")
    # v.set_current_neutral_face()
    # v.animate_mesh()
    # v.stop()
    # a = AudioAnimation()
    # a.set_audio(f"{PROJECT_DIR}/other_data/input_audios/2.wav")
    # a.animate_mesh()
    va = Animation(audio_support_level=0.8)
    va.set_parameters(video_path=f"{PROJECT_DIR}/audio_animation/dataset/raw_data/GH035257_52.mp4")
    va.animate_mesh()
