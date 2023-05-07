import moviepy.editor as mpe


def add_audio_to_video(input_video_path, audio_path, output_video_path, remove_original=True):
    video_stream = mpe.VideoFileClip(input_video_path, audio=False)
    audio_stream = mpe.AudioFileClip(audio_path)
    final_clip = video_stream.set_audio(audio_stream)
    final_clip.write_videofile(output_video_path)
