from config.paths import PROJECT_DIR
import torch
from transformers import AutoModelForCTC
import librosa


class VoiceProcessor:
    def __init__(self, cuda=True):
        self.model = AutoModelForCTC.from_pretrained("UrukHan/wav2vec2-russian")
        if cuda:
            self.model = self.model.to("cuda")

    def execute(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000)
        audio = list(audio)
        with torch.no_grad():
            input_values = torch.tensor(audio, device="cuda").unsqueeze(0)
            return self.model(input_values).logits


if __name__ == "__main__":
    vp = VoiceProcessor()
    output = vp.execute(f"{PROJECT_DIR}/other_data/input_audios/1.wav")
