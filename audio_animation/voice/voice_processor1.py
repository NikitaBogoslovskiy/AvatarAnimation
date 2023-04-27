from deepspeech import Model
import wave
import numpy as np
import subprocess
import shlex
try:
    from shhlex import quote
except ImportError:
    from pipes import quote
import librosa
from librosa.feature import mfcc
import soundfile as sf
import torchaudio.models


class VoiceProcessor:
    def __init__(self):
        self.model_path = "C:/Content/Python/AvatarAnimation/audio_animation/voice/deepspeech_files/" \
                           "deepspeech-0.9.3-models.pbmm"
        self.scorer_path = "C:/Content/Python/AvatarAnimation/audio_animation/voice/deepspeech_files/" \
                            "deepspeech-0.9.3-models.scorer"
        self.model = Model(self.model_path)
        self.model.enableExternalScorer(self.scorer_path)
        self.model_frame_rate = self.model.sampleRate()
        # self.model = torchaudio.models.DeepSpeech(n_feature=274)

    def get_features_from_audio(self, audio_path):
        audio, frame_rate = librosa.load(audio_path, sr=self.model_frame_rate)
        mfccs = mfcc(y=audio, sr=frame_rate, n_mfcc=40)
        print(mfccs.shape)
        # sf.write('new_test.wav', audio, sr, 'PCM_24')
        audio = audio.astype(np.int16)
        # audio_stream = wave.open(audio_path, 'rb')
        # audio_frame_rate = audio_stream.getframerate()
        # model_frame_rate = self.model.sampleRate()
        # if audio_frame_rate != model_frame_rate:
        #     audio = self._convert_sample_rate(audio_path, model_frame_rate)
        # else:
        #     audio = audio_stream.readframes(audio_stream.getnframes())
        # audio = np.frombuffer(audio, np.int16)
        # audio_length = audio_stream.getnframes() * (1 / audio_frame_rate)
        # audio_stream.close()
        output = self.model.stt(audio)
        print(output)

    def _convert_sample_rate(self, audio_path, desired_sample_rate):
        sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little ' \
                  '--compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
        try:
            output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
        except OSError as e:
            raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

        return output


if __name__ == "__main__":
    vp = VoiceProcessor()
    vp.get_features_from_audio("test4.wav")
