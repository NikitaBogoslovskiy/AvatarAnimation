from config.paths import PROJECT_DIR
from audio_animation.model.audio_model_pytorch import AudioModelPyTorch
from utils.torch_funcs import init_weights
from os import walk
import torch
import random
from audio_animation.dataset.dataset import Dataset
from FLAME.flame_model import FlameModel
from FLAME.utils import upload_face_mask, upload_lips_mask
from FLAME.config import get_config
from progress.bar import Bar
from utils.progress_bar import TrainingBar
from datetime import datetime
from FLAME.flame_model import RADIAN


class AudioModelTrainParams:
    def __init__(self,
                 dataset_path,
                 output_weights_path,
                 train_percentage=0.95,
                 epoch_number=1,
                 model_batch_size=2,
                 flame_batch_size=75,
                 sequence_length=375,
                 learning_rate=1e-3,
                 decay_rate=0.98,
                 decay_frequency=10000,
                 correctness_coefficient=1.0,
                 smoothing_coefficient=0.5,
                 weight_decay=0.0):
        if flame_batch_size > sequence_length:
            raise Exception("flame_batch_size must be less or equal than sequence length")
        if sequence_length % flame_batch_size != 0:
            raise Exception("Remainder of division of sequence length by flame_batch_size must be equal 0")
        self.dataset_path = dataset_path
        self.output_weights_path = output_weights_path
        self.train_percentage = train_percentage
        self.epoch_number = epoch_number
        self.model_batch_size = model_batch_size
        self.flame_batch_size = flame_batch_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_frequency = decay_frequency
        self.correctness_coefficient = correctness_coefficient
        self.smoothing_coefficient = smoothing_coefficient
        self.weight_decay = weight_decay


class AudioModelExecuteParams:
    def __init__(self,
                 audio_features=None,
                 expr_min=-2.3,
                 expr_max=2.3,
                 jaw_min=0.0,
                 jaw_max=20 * RADIAN):
        self.audio_features = audio_features
        self.expr_min = expr_min
        self.expr_max = expr_max
        self.jaw_min = jaw_min
        self.jaw_max = jaw_max


class AudioModel:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.initialized_for_execution = False
        self.torch_model = None
        self.flame_model = None
        self.default_shape = None
        self.default_position = None
        self.default_jaw = None
        self.neutral_vertices = None
        self.neutral_landmarks = None
        self.flame_batch_size = None
        self.face_mask = upload_face_mask()
        self.lips_mask = upload_lips_mask()
        if self.cuda:
            self.face_mask = self.face_mask.cuda()
            self.lips_mask = self.lips_mask.cuda()

    def load_model(self, weights_path):
        self.torch_model = AudioModelPyTorch()
        self.torch_model.load_state_dict(torch.load(weights_path))
        self.torch_model.eval()
        if self.cuda:
            self.torch_model.to(torch.device("cuda"))

    def init_for_execution(self, flame_batch_size):
        self.flame_batch_size = flame_batch_size
        self.flame_model = FlameModel(get_config(flame_batch_size), self.cuda)
        self.neutral_vertices, _ = Dataset.upload_neutral(
            f"{PROJECT_DIR}/video_animation/dataset/train_data/neutral.json")
        self.neutral_vertices = torch.Tensor(self.neutral_vertices)
        self.default_shape = torch.zeros(flame_batch_size, 100)
        self.default_position = torch.zeros(flame_batch_size, 3)
        self.default_jaw = torch.zeros(flame_batch_size, 2)
        if self.cuda:
            self.default_shape = self.default_shape.cuda()
            self.default_position = self.default_position.cuda()
            self.default_jaw = self.default_jaw.cuda()
            self.neutral_vertices = self.neutral_vertices.cuda()
        self.initialized_for_execution = True

    def execute(self, params: AudioModelExecuteParams):
        if self.torch_model is None:
            raise Exception("You cannot execute uninitialized model. Load the model.")
        if not self.init_for_execution:
            raise Exception("You have to call init_for_execution once before execution.")
        output = self.torch_model(params.audio_features)
        num_items = output.size(dim=0)
        current_batch_size = self.flame_batch_size
        num_batches = num_items // self.flame_batch_size
        iterations_number = num_batches + 1 if num_items % self.flame_batch_size != 0 else num_batches
        output_vertices = self.neutral_vertices[None, :].repeat(self.flame_batch_size, 1, 1)
        expressions = torch.zeros(self.flame_batch_size, 100)
        jaw = torch.zeros(self.flame_batch_size, 1)
        output[:, :100] = torch.clip(output[:, :100], params.expr_min, params.expr_max)
        output[:, 100] = torch.clip(output[:, 100], params.jaw_min, params.jaw_max)
        for batch_idx in range(iterations_number):
            if batch_idx == num_batches:
                current_batch_size = num_items - num_batches * self.flame_batch_size
            expressions[:current_batch_size] = output[batch_idx * self.flame_batch_size: batch_idx * self.flame_batch_size + current_batch_size, :100]
            jaw[:current_batch_size] = output[batch_idx * self.flame_batch_size: batch_idx * self.flame_batch_size + current_batch_size, 100][:, None]
            head_vertices, _ = self.flame_model.generate(
                self.default_shape, torch.cat([self.default_position, jaw, self.default_jaw], dim=1), expressions)
            output_vertices[:, self.face_mask] = head_vertices[:, self.face_mask]
            yield current_batch_size, output_vertices.cpu()
        yield None, None

    @staticmethod
    def _normalize_sequence_length(lips_positions, audio_features):
        max_indices = torch.argmax(audio_features, dim=1)
        non_blank_indices = (max_indices != 0).nonzero(as_tuple=True)
        start_idx, end_idx = non_blank_indices[0], non_blank_indices[-1]
        audio_features_length = len(audio_features)
        blank_overall_length = start_idx + (audio_features_length - end_idx - 1)
        if audio_features_length > params.sequence_length:
            extra_length = audio_features_length - params.sequence_length
            if blank_overall_length > extra_length:
                start_blank_percentage, end_blank_percentage = start_idx / blank_overall_length, (audio_features_length - end_idx - 1) / blank_overall_length
                new_start_idx, new_end_idx = int(round(start_blank_percentage * extra_length)), audio_features_length - int(round(end_blank_percentage * extra_length))
            elif blank_overall_length < extra_length:
                new_start_idx, new_end_idx = start_idx, end_idx - (extra_length - blank_overall_length)
            else:
                new_start_idx, new_end_idx = start_idx, end_idx
            return lips_positions[new_end_idx: new_end_idx], audio_features[new_end_idx: new_end_idx]
        elif audio_features_length < params.sequence_length:
            missing_length = params.sequence_length - audio_features_length
            start_blank_percentage, end_blank_percentage = start_idx / blank_overall_length, (audio_features_length - end_idx - 1) / blank_overall_length
            start_dummies_number, end_dummies_number = int(round(start_blank_percentage * missing_length)), audio_features_length - int(round(end_blank_percentage * missing_length))
            start_insertion_step, end_insertion_step = int(round(start_idx / start_dummies_number)), int(round((audio_features_length - end_idx - 1) / end_dummies_number))
            new_lips_list, new_audio_list = [], []
            for start_dummy_idx in range(start_dummies_number):
                if start_dummy_idx != start_dummies_number - 1:
                    slice_idx_start, slice_idx_end = start_dummy_idx * start_insertion_step, (start_dummy_idx + 1) * start_insertion_step
                else:
                    slice_idx_start, slice_idx_end = start_dummy_idx * start_insertion_step, start_idx
                insertion_idx = slice_idx_end - 1
                new_lips_list.append(lips_positions[slice_idx_start: slice_idx_end])
                new_lips_list.append(lips_positions[insertion_idx])
                new_audio_list.append(audio_features[slice_idx_start: slice_idx_end])
                new_audio_list.append(audio_features[insertion_idx])
            new_lips_list.append(lips_positions[start_idx: end_idx + 1])
            new_audio_list.append(audio_features[start_idx: end_idx + 1])
            for end_dummy_idx in range(end_dummies_number):
                if end_dummy_idx != end_dummies_number - 1:
                    slice_idx_start, slice_idx_end = (end_idx + 1) + end_dummy_idx * end_insertion_step, (end_idx + 1) + (end_dummy_idx + 1) * end_insertion_step
                else:
                    slice_idx_start, slice_idx_end = (end_idx + 1) + end_dummy_idx * end_insertion_step, audio_features_length
                insertion_idx = slice_idx_end - 1
                new_lips_list.append(lips_positions[slice_idx_start: slice_idx_end])
                new_lips_list.append(lips_positions[insertion_idx])
                new_audio_list.append(audio_features[slice_idx_start: slice_idx_end])
                new_audio_list.append(audio_features[insertion_idx])
            return torch.stack(new_lips_list), torch.stack(new_audio_list)

    def train(self, params: AudioModelTrainParams):
        if self.torch_model is None:
            self.torch_model = AudioModelPyTorch()
            self.torch_model.apply(init_weights)
        if self.cuda:
            self.torch_model.to(torch.device("cuda"))
        train_flame_model = FlameModel(get_config(params.flame_batch_size), self.cuda)

        item_names = next(walk(params.dataset_path), (None, None, []))[2]
        item_names = list(filter(lambda x: x.endswith(".json"), item_names))
        random.shuffle(item_names)
        num_items = len(item_names)
        num_batches = num_items // params.model_batch_size
        num_train_batches = int(num_batches * params.train_percentage)
        num_test_batches = num_batches - num_train_batches
        num_flame_batches = params.sequence_length // params.flame_batch_size
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=params.learning_rate,
                                     weight_decay=params.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=params.decay_rate)
        default_shape = torch.zeros(params.flame_batch_size, 100)
        default_position = torch.zeros(params.flame_batch_size, 3)
        default_jaw = torch.zeros(params.flame_batch_size, 2)
        if self.cuda:
            default_shape = default_shape.cuda()
            default_position = default_position.cuda()
            default_jaw = default_jaw.cuda()

        bar = TrainingBar('Training progress', max=num_train_batches * params.model_batch_size * params.epoch_number,
                          check_tty=False)
        counter = 0
        print("Started training")
        for epoch_idx in range(params.epoch_number):
            for batch_idx in range(num_train_batches):
                ground_truth_lips_positions, input_audio_features = [], []
                for item_idx in range(params.model_batch_size):
                    lips_positions, audio_features = Dataset.upload(f"{params.dataset_path}/{item_names[batch_idx * params.model_batch_size + item_idx]}")
                    lips_positions = torch.Tensor(lips_positions)
                    audio_features = torch.Tensor(audio_features)
                    if len(audio_features) != params.sequence_length:
                        lips_positions, audio_features = AudioModel._normalize_sequence_length(lips_positions, audio_features)
                    ground_truth_lips_positions.append(lips_positions)
                    input_audio_features.append(audio_features)
                ground_truth_lips_positions = torch.stack(ground_truth_lips_positions)
                input_audio_features = torch.stack(input_audio_features)
                if self.cuda:
                    ground_truth_lips_positions = ground_truth_lips_positions.cuda()
                    input_audio_features = input_audio_features.cuda()
                optimizer.zero_grad()
                output = self.torch_model(input_audio_features)
                output_lips_positions = []
                for item_idx in range(params.model_batch_size):
                    local_output_lips_positions = []
                    for flame_batch_idx in range(num_flame_batches):
                        start_idx, end_idx = flame_batch_idx * params.flame_batch_size, (flame_batch_idx + 1) * params.flame_batch_size
                        head_vertices, _ = train_flame_model.generate(default_shape, torch.cat([default_position, output[item_idx, start_idx: end_idx, 100][:, None], default_jaw], dim=1),
                                                                      output[item_idx, start_idx: end_idx, :100])
                        local_output_lips_positions.append(head_vertices[self.lips_mask])
                    output_lips_positions.append(torch.stack(local_output_lips_positions))
                output_lips_positions = torch.stack(output_lips_positions)
                loss = params.correctness_coefficient * torch.mean(torch.sum(torch.sum(torch.norm(ground_truth_lips_positions - output_lips_positions, dim=3), dim=2), dim=1)) + \
                       params.smoothing_coefficient * torch.mean(torch.sum(torch.sum(torch.norm((ground_truth_lips_positions[:, 1:] - ground_truth_lips_positions[:, :-1])
                                                                                                - (output_lips_positions[:, 1:] - output_lips_positions[:, :-1])))))
                bar.set_loss(loss)
                loss.backward()
                optimizer.step()
                bar.next(params.model_batch_size)
                if counter > params.decay_frequency:
                    lr_scheduler.step()
                    counter = 0
        bar.finish()
        print("Finished training")

        test_start = num_train_batches * params.model_batch_size
        loss = 0
        bar = Bar('Testing progress', max=num_test_batches * params.model_batch_size, check_tty=False)
        print("Started evaluating test data")
        for batch_idx in range(num_test_batches):
            ground_truth_lips_positions, input_audio_features = [], []
            for item_idx in range(params.model_batch_size):
                lips_positions, audio_features = Dataset.upload(f"{params.dataset_path}/{item_names[test_start + batch_idx * params.model_batch_size + item_idx]}")
                lips_positions = torch.Tensor(lips_positions)
                audio_features = torch.Tensor(audio_features)
                if len(audio_features) != params.sequence_length:
                    lips_positions, audio_features = AudioModel._normalize_sequence_length(lips_positions, audio_features)
                ground_truth_lips_positions.append(lips_positions)
                input_audio_features.append(audio_features)
            ground_truth_lips_positions = torch.stack(ground_truth_lips_positions)
            input_audio_features = torch.stack(input_audio_features)
            if self.cuda:
                ground_truth_lips_positions = ground_truth_lips_positions.cuda()
                input_audio_features = input_audio_features.cuda()
            optimizer.zero_grad()
            output = self.torch_model(input_audio_features)
            output_lips_positions = []
            for item_idx in range(params.model_batch_size):
                local_output_lips_positions = []
                for flame_batch_idx in range(num_flame_batches):
                    start_idx, end_idx = flame_batch_idx * params.flame_batch_size, (flame_batch_idx + 1) * params.flame_batch_size
                    lips_vertices, _ = train_flame_model.generate(default_shape, torch.cat([default_position, output[item_idx, start_idx: end_idx, 100][:, None], default_jaw], dim=1),
                                                                  output[item_idx, start_idx: end_idx, :100])
                    local_output_lips_positions.append(lips_vertices)
                output_lips_positions.append(torch.stack(local_output_lips_positions))
            output_lips_positions = torch.stack(output_lips_positions)
            loss += params.correctness_coefficient * torch.mean(torch.sum(torch.sum(torch.norm(ground_truth_lips_positions - output_lips_positions, dim=3), dim=2), dim=1)) + \
                   params.smoothing_coefficient * torch.mean(torch.sum(torch.sum(torch.norm((ground_truth_lips_positions[:, 1:] - ground_truth_lips_positions[:, :-1])
                                                                                            - (output_lips_positions[:, 1:] - output_lips_positions[:, :-1])))))
            optimizer.step()
            bar.next(params.model_batch_size)

        bar.finish()
        print("Finished testing")
        print(f"Test loss = {loss / num_test_batches:.4f}")

        torch.save(self.torch_model.state_dict(),
                   f"{params.output_weights_path}/video_model_{params.epoch_number}_"
                   f"{num_train_batches * params.model_batch_size}_"
                   f"{datetime.now().strftime('%d.%m.%Y-%H.%M.%S')}.pt")
        print(f"Model has been saved to {params.output_weights_path}")


if __name__ == "__main__":
    pass
    params = AudioModelTrainParams(
        dataset_path=f"{PROJECT_DIR}/audio_animation/dataset/train_data",
        output_weights_path=f"{PROJECT_DIR}/audio_animation/weights",
        train_percentage=0.95,
        epoch_number=1,
        model_batch_size=2,
        flame_batch_size=75,
        sequence_length=375,
        learning_rate=1e-3,
        decay_rate=0.98,
        decay_frequency=10000,
        correctness_coefficient=1.0,
        smoothing_coefficient=0.5,
        weight_decay=0.0
    )
    audio_model = AudioModel(cuda=True)
    audio_model.train(params)
