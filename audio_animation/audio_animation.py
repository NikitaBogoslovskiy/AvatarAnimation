from config.paths import PROJECT_DIR
from video_animation.visualizer.offline_visualizer import OfflineVisualizer
import os
from utils.video_functions import add_audio_to_video
from progress.bar import Bar
from audio_animation.model.audio_model import AudioModel, AudioModelExecuteParams
from audio_animation.wav2vec2.voice_processor import VoiceProcessor


class AudioAnimation:
    def __init__(self, cuda=True, logging=True):
        self.logging = logging
        if self.logging:
            print("Preparing for animation... ", end='')
        self.visualizer = None
        self.input_audio_path = None
        self.voice_processor = VoiceProcessor(cuda)
        self.cuda = cuda
        self.audio_model = AudioModel(self.cuda)
        self.audio_model.load_model(weights_path=f"{PROJECT_DIR}/audio_animation/weights/audio_model_weights.pt")
        self.output_video_path_without_audio = None
        self.output_video_path_with_audio = None
        self.audio_path = None
        self.execution_params = AudioModelExecuteParams()

    def set_audio(self, audio_path, audio_features=None):
        if audio_features is None:
            if not os.path.isfile(audio_path):
                raise FileNotFoundError("File path is incorrect")
            self.execution_params.audio_features = self.voice_processor.execute(audio_path)
        else:
            self.execution_params.audio_features = audio_features
        self.audio_path = audio_path
        if self.cuda:
            self.execution_params.audio_features = self.execution_params.audio_features.cuda()
        self.audio_model.init_for_execution(flame_batch_size=200)

    def _init_visualizer(self, output_resolution):
        directory = os.path.dirname(self.audio_path)
        audio_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        self.input_audio_path = self.audio_path
        self.output_video_path_without_audio = f"{directory}/output_video_stream.mp4"
        self.output_video_path_with_audio = f"{directory}/{audio_name}.mp4"
        self.visualizer = OfflineVisualizer(self.output_video_path_without_audio)
        self.visualizer.init_settings(animation_resolution=output_resolution, input_resolution=None, frame_rate=50)
        self.visualizer.set_surfaces(self.audio_model.flame_model.flamelayer.faces)

    def _release_visualizer(self):
        self.visualizer.release()
        add_audio_to_video(input_video_path=self.output_video_path_without_audio, audio_path=self.input_audio_path, output_video_path=self.output_video_path_with_audio)
        os.remove(self.output_video_path_without_audio)
        if self.logging:
            print(f"Output animation video has been saved to '{self.output_video_path_with_audio}'")

    def animate_mesh(self, resolution=(512, 512)):
        self._init_visualizer(resolution)
        processed_features = self.audio_model.execute(self.execution_params)
        if self.logging:
            print("Done.")
            bar = Bar('Audio processing', max=self.execution_params.audio_features.size()[1], check_tty=False)
        while True:
            current_batch_size, output_vertices = next(processed_features)
            if current_batch_size is None:
                break
            output_vertices = output_vertices.numpy().squeeze()
            for idx in range(current_batch_size):
                self.visualizer.render(output_vertices[idx])
                if self.logging:
                    bar.next()
        if self.logging:
            bar.finish()
        self._release_visualizer()
