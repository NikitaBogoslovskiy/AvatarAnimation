from datetime import datetime

from config.paths import PROJECT_DIR
from video_animation.model.video_model_pytorch import VideoModelPyTorch
from utils.torch_funcs import init_weights
from os import walk
import torch
import random
from video_animation.dataset.dataset import Dataset
from utils.landmarks import divide_landmarks, FACIAL_LANDMARKS, MOUTH_LANDMARKS, LEFT_EYE_LANDMARKS, LEFT_EYEBROW_LANDMARKS, RIGHT_EYE_LANDMARKS, RIGHT_EYEBROW_LANDMARKS, NOSE_LANDMARKS, JAW_LANDMARKS
from FLAME.flame_model import FlameModel
from FLAME.utils import upload_face_mask, upload_lips_mask, upload_masks
from FLAME.config import get_config
from progress.bar import Bar
from utils.progress_bar import TrainingBar
import string
from FLAME.flame_model import RADIAN


class VideoModelTrainParams:
    def __init__(self,
                 dataset_path,
                 output_weights_path,
                 train_percentage=0.95,
                 epoch_number=1,
                 batch_size=500,
                 learning_rate=1e-3,
                 decay_rate=0.98,
                 decay_frequency=10000,
                 noise_level=1e-3,
                 regularization=5e-4,
                 weight_decay=0.0,
                 left_eye_landmarks_loss_coefficient=1.0,
                 right_eye_landmarks_loss_coefficient=1.0,
                 nose_landmarks_loss_coefficient=1.0,
                 lips_landmarks_loss_coefficient=1.0,
                 jaw_landmarks_loss_coefficient=1.0,
                 forehead_vertices_loss_coefficient=1.0,
                 left_eye_vertices_loss_coefficient=1.0,
                 right_eye_vertices_loss_coefficient=1.0,
                 nose_vertices_loss_coefficient=1.0,
                 lips_vertices_loss_coefficient=1.0):
        self.dataset_path = dataset_path
        self.output_weights_path = output_weights_path
        self.train_percentage = train_percentage
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_frequency = decay_frequency
        self.noise_level = noise_level
        self.regularization = regularization
        self.weight_decay = weight_decay
        self.left_eye_landmarks_loss_coefficient = left_eye_landmarks_loss_coefficient
        self.right_eye_landmarks_loss_coefficient = right_eye_landmarks_loss_coefficient
        self.nose_landmarks_loss_coefficient = nose_landmarks_loss_coefficient
        self.lips_landmarks_loss_coefficient = lips_landmarks_loss_coefficient
        self.jaw_landmarks_loss_coefficient = jaw_landmarks_loss_coefficient
        self.forehead_vertices_loss_coefficient = forehead_vertices_loss_coefficient
        self.left_eye_vertices_loss_coefficient = left_eye_vertices_loss_coefficient
        self.right_eye_vertices_loss_coefficient = right_eye_vertices_loss_coefficient
        self.nose_vertices_loss_coefficient = nose_vertices_loss_coefficient
        self.lips_vertices_loss_coefficient = lips_vertices_loss_coefficient


class VideoModelExecuteParams:
    def __init__(self,
                 left_eye=None,
                 right_eye=None,
                 nose_mouth=None,
                 expr_min=-2.5,
                 expr_max=2.5,
                 jaw_min=0.0,
                 jaw_max=2.5 * RADIAN):
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.nose_mouth = nose_mouth
        self.expr_min = expr_min
        self.expr_max = expr_max
        self.jaw_min = jaw_min
        self.jaw_max = jaw_max


class VideoModel:
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
        self.landmarks_output = None
        self.face_mask = upload_face_mask()
        self.lips_mask = upload_lips_mask()
        self.masks = upload_masks()
        if self.cuda:
            self.face_mask = self.face_mask.cuda()
            self.lips_mask = self.lips_mask.cuda()
            self.masks.to_cuda()

    def load_model(self, weights_path):
        self.torch_model = VideoModelPyTorch()
        self.torch_model.load_state_dict(torch.load(weights_path))
        self.torch_model.eval()
        if self.cuda:
            self.torch_model.to(torch.device("cuda"))

    def init_for_execution(self, batch_size):
        self.flame_model = FlameModel(get_config(batch_size), self.cuda)
        self.default_shape = torch.zeros(batch_size, 100)
        self.default_position = torch.zeros(batch_size, 3)
        self.default_jaw = torch.zeros(batch_size, 2)
        self.neutral_vertices, self.neutral_landmarks = Dataset.upload_neutral(
            f"{PROJECT_DIR}/video_animation/dataset/train_data/neutral.json")
        self.neutral_vertices = torch.Tensor(self.neutral_vertices)
        self.neutral_landmarks = torch.Tensor(self.neutral_landmarks)
        if self.cuda:
            self.default_shape = self.default_shape.cuda()
            self.default_position = self.default_position.cuda()
            self.default_jaw = self.default_jaw.cuda()
            self.neutral_vertices = self.neutral_vertices.cuda()
            self.neutral_landmarks = self.neutral_landmarks.cuda()
        self.initialized_for_execution = True

    def execute(self, params: VideoModelExecuteParams):
        if self.torch_model is None:
            raise Exception("You cannot execute uninitialized model. Load the model.")
        if not self.init_for_execution:
            raise Exception("You have to call init_for_execution once before execution.")
        output = self.torch_model(params.left_eye, params.right_eye, params.nose_mouth)
        expressions = torch.clip(output[:, :100], params.expr_min, params.expr_max)
        jaw = torch.clip(output[:, 100], params.jaw_min, params.jaw_max)[:, None]
        generated_vertices, self.landmarks_output = self.flame_model.generate(
            self.default_shape, torch.cat([self.default_position, jaw, self.default_jaw], dim=1), expressions)
        return generated_vertices.detach()

    def train(self, params: VideoModelTrainParams):
        if self.torch_model is None:
            self.torch_model = VideoModelPyTorch()
            self.torch_model.apply(init_weights)
        if self.cuda:
            self.torch_model.to(torch.device("cuda"))
        train_flame_model = FlameModel(get_config(params.batch_size), self.cuda)

        item_names = next(walk(params.dataset_path), (None, None, []))[2]
        item_names_set = set(item_names)
        item_names_set.remove("neutral.json")
        item_names = list(item_names_set)
        del item_names_set
        random.shuffle(item_names)
        num_items = len(item_names)
        num_batches = num_items // params.batch_size
        num_train_batches = int(num_batches * params.train_percentage)
        num_test_batches = num_batches - num_train_batches
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=params.learning_rate,
                                     weight_decay=params.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=params.decay_rate)
        default_shape = torch.zeros(params.batch_size, 100)
        default_position = torch.zeros(params.batch_size, 3)
        default_jaw = torch.zeros(params.batch_size, 2)
        if self.cuda:
            default_shape = default_shape.cuda()
            default_position = default_position.cuda()
            default_jaw = default_jaw.cuda()

        bar = TrainingBar('Training progress', max=num_train_batches * params.batch_size * params.epoch_number,
                          check_tty=False)
        counter = 0
        print("Started training")
        for epoch_idx in range(params.epoch_number):
            for batch_idx in range(num_train_batches):
                noise = torch.rand((params.batch_size, 54, 2)) * params.noise_level
                left_eyes, right_eyes, noses_mouths = [], [], []
                origin_forehead, origin_left_eye_region, origin_right_eye_region, origin_nose, origin_lips, origin_landmarks = [], [], [], [], [], []
                for item_idx in range(params.batch_size):
                    data_item = Dataset.upload(f"{params.dataset_path}/{item_names[batch_idx * params.batch_size + item_idx]}")
                    origin_forehead.append(torch.tensor(data_item["forehead"]))
                    origin_left_eye_region.append(torch.tensor(data_item["left_eye_region"]))
                    origin_right_eye_region.append(torch.tensor(data_item["right_eye_region"]))
                    origin_nose.append(torch.tensor(data_item["nose"]))
                    origin_lips.append(torch.tensor(data_item["lips"]))
                    landmarks_tensor = torch.tensor(data_item["landmarks"])
                    origin_landmarks.append(landmarks_tensor)
                    left_eye, right_eye, nose_mouth = divide_landmarks(landmarks_tensor)
                    left_eyes.append(left_eye)
                    right_eyes.append(right_eye)
                    noses_mouths.append(nose_mouth)
                    counter += 1
                origin_forehead = torch.stack(origin_forehead)
                origin_left_eye_region = torch.stack(origin_left_eye_region)
                origin_right_eye_region = torch.stack(origin_right_eye_region)
                origin_nose = torch.stack(origin_nose)
                origin_lips = torch.stack(origin_lips)
                origin_landmarks = torch.stack(origin_landmarks)
                left_eyes = torch.stack(left_eyes)[:, :, :2] + noise[:, :11]
                right_eyes = torch.stack(right_eyes)[:, :, :2] + noise[:, 11:22]
                noses_mouths = torch.stack(noses_mouths)[:, :, :2] + noise[:, 22:]
                if self.cuda:
                    origin_forehead = origin_forehead.cuda()
                    origin_left_eye_region = origin_left_eye_region.cuda()
                    origin_right_eye_region = origin_right_eye_region.cuda()
                    origin_nose = origin_nose.cuda()
                    origin_lips = origin_lips.cuda()
                    origin_landmarks = origin_landmarks.cuda()
                    left_eyes = left_eyes.cuda()
                    right_eyes = right_eyes.cuda()
                    noses_mouths = noses_mouths.cuda()
                optimizer.zero_grad()
                output = self.torch_model(left_eyes, right_eyes, noses_mouths)
                generated_vertices, generated_landmarks = train_flame_model.generate(
                    default_shape, torch.cat([default_position,
                                              output[:, 100][:, None], default_jaw], dim=1), output[:, :100])
                landmarks_loss = params.left_eye_landmarks_loss_coefficient * torch.mean(torch.sum(
                    torch.linalg.norm(origin_landmarks[:, [*LEFT_EYE_LANDMARKS, *LEFT_EYEBROW_LANDMARKS]] - generated_landmarks[:, [*LEFT_EYE_LANDMARKS, *LEFT_EYEBROW_LANDMARKS]], dim=2), dim=1)) + \
                                 params.right_eye_landmarks_loss_coefficient * torch.mean(torch.sum(
                    torch.linalg.norm(origin_landmarks[:, [*RIGHT_EYE_LANDMARKS, *RIGHT_EYEBROW_LANDMARKS]] - generated_landmarks[:, [*RIGHT_EYE_LANDMARKS, *RIGHT_EYEBROW_LANDMARKS]], dim=2),dim=1)) + \
                                 params.nose_landmarks_loss_coefficient * torch.mean(torch.sum(
                    torch.linalg.norm(origin_landmarks[:, NOSE_LANDMARKS] - generated_landmarks[:, NOSE_LANDMARKS], dim=2), dim=1)) + \
                                 params.lips_landmarks_loss_coefficient * torch.mean(torch.sum(
                    torch.linalg.norm(origin_landmarks[:, MOUTH_LANDMARKS] - generated_landmarks[:, MOUTH_LANDMARKS], dim=2), dim=1)) + \
                                 params.jaw_landmarks_loss_coefficient * torch.mean(torch.sum(
                    torch.linalg.norm(origin_landmarks[:, JAW_LANDMARKS] - generated_landmarks[:, JAW_LANDMARKS], dim=2), dim=1))

                vertices_loss = params.forehead_vertices_loss_coefficient * torch.mean(torch.sum(
                    torch.linalg.norm(origin_forehead - generated_vertices[:, self.masks.forehead], dim=2), dim=1)) + \
                                params.left_eye_vertices_loss_coefficient * torch.mean(torch.sum(
                    torch.linalg.norm(origin_left_eye_region - generated_vertices[:, self.masks.left_eye_region], dim=2), dim=1)) + \
                                params.right_eye_vertices_loss_coefficient * torch.mean(torch.sum(
                    torch.linalg.norm(origin_right_eye_region - generated_vertices[:, self.masks.right_eye_region], dim=2), dim=1)) + \
                                params.nose_vertices_loss_coefficient * torch.mean(torch.sum(
                    torch.linalg.norm(origin_nose - generated_vertices[:, self.masks.nose], dim=2), dim=1)) + \
                                params.lips_vertices_loss_coefficient * torch.mean(torch.sum(
                    torch.linalg.norm(origin_lips - generated_vertices[:, self.masks.lips], dim=2), dim=1))

                overall_loss = landmarks_loss + vertices_loss + params.regularization * torch.linalg.norm(output)
                bar.set_loss(overall_loss)
                overall_loss.backward()
                optimizer.step()
                bar.next(params.batch_size)
                if counter > params.decay_frequency:
                    lr_scheduler.step()
                    counter = 0
        bar.finish()
        print("Finished training")

        test_start = num_train_batches * params.batch_size
        overall_loss = 0
        bar = Bar('Testing progress', max=num_test_batches * params.batch_size, check_tty=False)
        print("Started evaluating test data")
        for batch_idx in range(num_test_batches):
            left_eyes, right_eyes, noses_mouths = [], [], []
            origin_forehead, origin_left_eye_region, origin_right_eye_region, origin_nose, origin_lips, origin_landmarks = [], [], [], [], [], []
            for item_idx in range(params.batch_size):
                data_item = Dataset.upload(f"{params.dataset_path}/{item_names[test_start + batch_idx * params.batch_size + item_idx]}")
                origin_forehead.append(torch.tensor(data_item["forehead"]))
                origin_left_eye_region.append(torch.tensor(data_item["left_eye_region"]))
                origin_right_eye_region.append(torch.tensor(data_item["right_eye_region"]))
                origin_nose.append(torch.tensor(data_item["nose"]))
                origin_lips.append(torch.tensor(data_item["lips"]))
                landmarks_tensor = torch.tensor(data_item["landmarks"])
                origin_landmarks.append(landmarks_tensor)
                left_eye, right_eye, nose_mouth = divide_landmarks(landmarks_tensor)
                left_eyes.append(left_eye)
                right_eyes.append(right_eye)
                noses_mouths.append(nose_mouth)
            origin_forehead = torch.stack(origin_forehead)
            origin_left_eye_region = torch.stack(origin_left_eye_region)
            origin_right_eye_region = torch.stack(origin_right_eye_region)
            origin_nose = torch.stack(origin_nose)
            origin_lips = torch.stack(origin_lips)
            origin_landmarks = torch.stack(origin_landmarks)
            left_eyes = torch.stack(left_eyes)[:, :, :2]
            right_eyes = torch.stack(right_eyes)[:, :, :2]
            noses_mouths = torch.stack(noses_mouths)[:, :, :2]
            if self.cuda:
                origin_forehead = origin_forehead.cuda()
                origin_left_eye_region = origin_left_eye_region.cuda()
                origin_right_eye_region = origin_right_eye_region.cuda()
                origin_nose = origin_nose.cuda()
                origin_lips = origin_lips.cuda()
                origin_landmarks = origin_landmarks.cuda()
                left_eyes = left_eyes.cuda()
                right_eyes = right_eyes.cuda()
                noses_mouths = noses_mouths.cuda()
            output = self.torch_model(left_eyes, right_eyes, noses_mouths)
            generated_vertices, generated_landmarks = train_flame_model.generate(
                default_shape, torch.cat([default_position,
                                          output[:, 100][:, None], default_jaw], dim=1), output[:, :100])
            landmarks_loss = params.left_eye_landmarks_loss_coefficient * torch.mean(torch.sum(
                torch.linalg.norm(origin_landmarks[:, [*LEFT_EYE_LANDMARKS, *LEFT_EYEBROW_LANDMARKS]] - generated_landmarks[:, [*LEFT_EYE_LANDMARKS, *LEFT_EYEBROW_LANDMARKS]], dim=2), dim=1)) + \
                             params.right_eye_landmarks_loss_coefficient * torch.mean(torch.sum(
                torch.linalg.norm(origin_landmarks[:, [*RIGHT_EYE_LANDMARKS, *RIGHT_EYEBROW_LANDMARKS]] - generated_landmarks[:, [*RIGHT_EYE_LANDMARKS, *RIGHT_EYEBROW_LANDMARKS]], dim=2), dim=1)) + \
                             params.nose_landmarks_loss_coefficient * torch.mean(torch.sum(
                torch.linalg.norm(origin_landmarks[:, NOSE_LANDMARKS] - generated_landmarks[:, NOSE_LANDMARKS], dim=2), dim=1)) + \
                             params.lips_landmarks_loss_coefficient * torch.mean(torch.sum(
                torch.linalg.norm(origin_landmarks[:, MOUTH_LANDMARKS] - generated_landmarks[:, MOUTH_LANDMARKS], dim=2), dim=1)) + \
                             params.jaw_landmarks_loss_coefficient * torch.mean(torch.sum(
                torch.linalg.norm(origin_landmarks[:, JAW_LANDMARKS] - generated_landmarks[:, JAW_LANDMARKS], dim=2), dim=1))

            vertices_loss = params.forehead_vertices_loss_coefficient * torch.mean(torch.sum(
                torch.linalg.norm(origin_forehead - generated_vertices[:, self.masks.forehead], dim=2), dim=1)) + \
                            params.left_eye_vertices_loss_coefficient * torch.mean(torch.sum(
                torch.linalg.norm(origin_left_eye_region - generated_vertices[:, self.masks.left_eye_region], dim=2), dim=1)) + \
                            params.right_eye_vertices_loss_coefficient * torch.mean(torch.sum(
                torch.linalg.norm(origin_right_eye_region - generated_vertices[:, self.masks.right_eye_region], dim=2), dim=1)) + \
                            params.nose_vertices_loss_coefficient * torch.mean(torch.sum(
                torch.linalg.norm(origin_nose - generated_vertices[:, self.masks.nose], dim=2), dim=1)) + \
                            params.lips_vertices_loss_coefficient * torch.mean(torch.sum(
                torch.linalg.norm(origin_lips - generated_vertices[:, self.masks.lips], dim=2), dim=1))

            overall_loss += landmarks_loss + vertices_loss + params.regularization * torch.linalg.norm(output)
            bar.next(params.batch_size)
        bar.finish()
        print("Finished testing")
        print(f"Test loss = {overall_loss / num_test_batches:.4f}")

        torch.save(self.torch_model.state_dict(),
                   f"{params.output_weights_path}/video_model_{params.epoch_number}_"
                   f"{num_train_batches * params.batch_size}_"
                   f"{datetime.now().strftime('%d.%m.%Y-%H.%M.%S')}.pt")
        print(f"Model has been saved to {params.output_weights_path}")


if __name__ == "__main__":
    pass
    params = VideoModelTrainParams(
        dataset_path=f"{PROJECT_DIR}/video_animation/dataset/train_data_4",
        output_weights_path=f"{PROJECT_DIR}/video_animation/weights",
        train_percentage=0.99,
        epoch_number=1,
        batch_size=200,
        learning_rate=1e-3,
        decay_rate=0.99,
        decay_frequency=10000,
        noise_level=1e-3,
        regularization=1e-4,
        weight_decay=0.0,
        left_eye_landmarks_loss_coefficient=1.0,
        right_eye_landmarks_loss_coefficient=1.0,
        nose_landmarks_loss_coefficient=1.0,
        lips_landmarks_loss_coefficient=1.0,
        jaw_landmarks_loss_coefficient=1.0,
        forehead_vertices_loss_coefficient=1.0,
        left_eye_vertices_loss_coefficient=1.0,
        right_eye_vertices_loss_coefficient=1.0,
        nose_vertices_loss_coefficient=1.0,
        lips_vertices_loss_coefficient=1.0
    )
    video_model = VideoModel(cuda=True)
    video_model.train(params)
