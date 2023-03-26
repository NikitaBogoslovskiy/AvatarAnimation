from animation_by_video.video_model_pytorch import VideoModelPyTorch
from utils.torch_funcs import init_weights
from os import walk
import torch
import random
from animation_by_video.dataset.dataset import Dataset
from utils.landmarks import divide_landmarks
from FLAME.flame_model import FlameModel
from FLAME.config import get_config
from progress.bar import Bar
from utils.progress_bar import TrainingBar


class VideoModelTrainParams:
    def __init__(self,
                 dataset_path,
                 output_weights_path,
                 train_percentage=0.95,
                 epoch_number=1,
                 batch_size=500,
                 learning_rate=1e-3,
                 decay_rate=0.98,
                 noise_level=1e-3,
                 regularization=5e-4,
                 weight_decay=0.0):
        self.dataset_path = dataset_path
        self.output_weights_path = output_weights_path
        self.train_percentage = train_percentage
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.noise_level = noise_level
        self.regularization = regularization
        self.weight_decay = weight_decay


class VideoModel:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.model = None

    def train(self, params: VideoModelTrainParams):
        self.model = VideoModelPyTorch()
        self.model.apply(init_weights)
        if self.cuda:
            self.model.to(torch.device("cuda"))
        flame_model = FlameModel(get_config(params.batch_size), self.cuda)

        item_names = next(walk(params.dataset_path), (None, None, []))[2]
        random.shuffle(item_names)
        num_items = len(item_names)
        num_batches = num_items // params.batch_size
        num_items = num_batches * params.batch_size
        num_train_batches = int(num_batches * params.train_percentage)
        num_test_batches = num_batches - num_train_batches
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=params.decay_rate)
        default_shape = torch.zeros(params.batch_size, 100)
        default_position = torch.zeros(params.batch_size, 3)
        if self.cuda:
            default_shape = default_shape.cuda()
            default_position = default_position.cuda()

        bar = TrainingBar('Training progress', max=num_train_batches * params.batch_size * params.epoch_number, check_tty=False)
        print("Started training")
        for epoch_idx in range(params.epoch_number):
            for batch_idx in range(num_train_batches):
                noise = torch.rand((params.batch_size, 51, 2)) * params.noise_level
                left_eyes, right_eyes, noses_mouths, origin_vertices = [], [], [], []
                for item_idx in range(params.batch_size):
                    vertices_i, landmarks_i = \
                        Dataset.upload(f"{params.dataset_path}/{item_names[batch_idx * params.batch_size + item_idx]}")
                    origin_vertices.append(torch.tensor(vertices_i))
                    left_eye, right_eye, nose_mouth = divide_landmarks(torch.tensor(landmarks_i))
                    left_eyes.append(left_eye)
                    right_eyes.append(right_eye)
                    noses_mouths.append(nose_mouth)
                origin_vertices = torch.stack(origin_vertices)
                left_eyes = torch.stack(left_eyes)[:, :, :2] + noise[:, :11, :]
                right_eyes = torch.stack(right_eyes)[:, :, :2] + noise[:, 11:22, :]
                noses_mouths = torch.stack(noses_mouths)[:, :, :2] + noise[:, 22:, :]
                if self.cuda:
                    origin_vertices = origin_vertices.cuda()
                    left_eyes = left_eyes.cuda()
                    right_eyes = right_eyes.cuda()
                    noses_mouths = noses_mouths.cuda()
                optimizer.zero_grad()
                output = self.model(left_eyes, right_eyes, noses_mouths)
                generated_vertices, _ = flame_model.generate(
                    default_shape, torch.cat([default_position, output[:, 50:]], dim=1), output[:, :50])
                loss = torch.mean(torch.sum(torch.linalg.norm(origin_vertices - generated_vertices, dim=2), dim=1))
                bar.set_loss(loss)
                loss.backward()
                optimizer.step()
                bar.next(params.batch_size)
                # print(f"epoch: {epoch_idx}, batch: {batch_idx}, loss = {loss}")
            lr_scheduler.step()
        bar.finish()
        print("Finished training")

        test_start = num_train_batches * params.batch_size
        loss = 0
        bar = Bar('Testing progress', max=num_test_batches * params.batch_size, check_tty=False)
        print("Started evaluating test data")
        for batch_idx in range(num_test_batches):
            left_eyes, right_eyes, noses_mouths, origin_vertices = [], [], [], []
            for item_idx in range(params.batch_size):
                vertices_i, landmarks_i = \
                    Dataset.upload(f"{params.dataset_path}/"
                                   f"{item_names[test_start + batch_idx * params.batch_size + item_idx]}")
                origin_vertices.append(torch.tensor(vertices_i))
                left_eye, right_eye, nose_mouth = divide_landmarks(torch.tensor(landmarks_i))
                left_eyes.append(left_eye)
                right_eyes.append(right_eye)
                noses_mouths.append(nose_mouth)
            origin_vertices = torch.stack(origin_vertices)
            left_eyes = torch.stack(left_eyes)[:, :, :2] + noise[:, :11, :]
            right_eyes = torch.stack(right_eyes)[:, :, :2] + noise[:, 11:22, :]
            noses_mouths = torch.stack(noses_mouths)[:, :, :2] + noise[:, 22:, :]
            if self.cuda:
                origin_vertices = origin_vertices.cuda()
                left_eyes = left_eyes.cuda()
                right_eyes = right_eyes.cuda()
                noses_mouths = noses_mouths.cuda()
            output = self.model(left_eyes, right_eyes, noses_mouths)
            generated_vertices, _ = flame_model.generate(
                default_shape, torch.cat([default_position, output[:, 50:]], dim=1), output[:, :50])
            loss += torch.mean(torch.sum(torch.linalg.norm(origin_vertices - generated_vertices, dim=2), dim=1))
            bar.next(params.batch_size)
        bar.finish()
        print("Finished testing")
        print(f"Test loss = {loss / num_test_batches:.4f}")

        torch.save(self.model.state_dict(),
                   f"{params.output_weights_path}/video_model_{params.epoch_number}_"
                   f"{num_train_batches * params.batch_size}.pt")
        print(f"Model has been saved to {params.output_weights_path}")


if __name__ == "__main__":
    params = VideoModelTrainParams(
        dataset_path="C:/Content/Python/AvatarAnimation/animation_by_video/dataset/train_data",
        output_weights_path="C:/Content/Python/AvatarAnimation/animation_by_video/weights",
        train_percentage=0.95,
        epoch_number=1,
        batch_size=100,
        learning_rate=1e-3,
        decay_rate=0.98,
        noise_level=1e-3,
        regularization=5e-4,
        weight_decay=0.0
    )
    video_model = VideoModel(cuda=True)
    video_model.train(params)
