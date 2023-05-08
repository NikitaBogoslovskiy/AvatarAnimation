from config.paths import PROJECT_DIR
import hydra
import torch
from audio_animation.deepspeech.deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser
from audio_animation.deepspeech.deepspeech_pytorch.utils import load_model
from audio_animation.deepspeech.deepspeech_pytorch.inference import get_raw_output
from transformers import AutoModelForCTC, Wav2Vec2Processor
import librosa


class VoiceProcessor:
    def __init__(self, cuda=True):
        # self.model_path = f"{PROJECT_DIR}/audio_animation/deepspeech/models/librispeech_pretrained_v3.ckpt"
        # self.cuda = cuda
        # self.device = torch.device("cuda" if self.cuda else "cpu")
        # self.model = load_model(
        #     device=self.device,
        #     model_path=self.model_path
        # )
        # self.spect_parser = ChunkSpectrogramParser(
        #     audio_conf=self.model.spect_cfg,
        #     normalize=True
        # )
        self.model1 = AutoModelForCTC.from_pretrained("UrukHan/wav2vec2-russian").to("cuda")

    def execute(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000)
        audio = list(audio)
        with torch.no_grad():
            input_values = torch.tensor(audio, device="cuda").unsqueeze(0)  # , device="cuda"
            output = self.model1(input_values).logits
        # pred_ids = torch.argmax(output, dim=-1)
        # output = get_raw_output(
        #     audio_path=hydra.utils.to_absolute_path(audio_path),
        #     spect_parser=self.spect_parser,
        #     model=self.model,
        #     device=self.device,
        #     precision=self.model.precision,
        #     chunk_size_seconds=-1.0
        # )
        return output


if __name__ == "__main__":
    vp = VoiceProcessor()
    output = vp.execute(f"{PROJECT_DIR}/audio_animation/deepspeech/test8.wav")
