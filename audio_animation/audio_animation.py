from audio_animation.model.audio_model import AudioModel, AudioModelExecuteParams
from config.paths import PROJECT_DIR
from audio_animation.deepspeech.voice_processor import VoiceProcessor
from video_animation.visualizer.offline_visualizer import OfflineVisualizer
import os


class AudioAnimation:
    def __init__(self, cuda=True):
        self.visualizer = None
        self.output_video_path = None
        self.voice_processor = VoiceProcessor()
        self.cuda = cuda
        self.audio_model = AudioModel(self.cuda)
        self.audio_model.load_model(weights_path=f"{PROJECT_DIR}/audio_animation/weights/video_model_1_96900_eaUV2Qs70Tlmn0S.pt")
        self.execution_params = AudioModelExecuteParams()

    def set_audio(self, audio_path, output_resolution=(512, 512)):
        if not os.path.isfile(audio_path):
            raise FileNotFoundError("File path is incorrect")
        self.execution_params.audio_features = self.voice_processor.execute(audio_path)
        directory = os.path.dirname(audio_path)
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        self.output_video_path = f"{directory}/{audio_name}.mp4"
        self.visualizer = OfflineVisualizer(self.output_video_path)
        self.visualizer.set_resolution(*output_resolution)
        self.audio_model.init_for_execution(flame_batch_size=200)
        self.visualizer.set_surfaces(self.audio_model.flame_model.flamelayer.faces)

    def animate_mesh(self):
        processed_features = self.audio_model.execute()
        while True:
            current_batch_size, output_vertices = next(processed_features)
            if current_batch_size is None:
                break
            output_vertices = output_vertices.numpy().squeeze()
            for idx in range(current_batch_size):
                self.visualizer.render(output_vertices[idx])
        print(f"Output animation video has been saved to '{self.output_video_path}'")

    def stop(self):
        self.visualizer.release()
