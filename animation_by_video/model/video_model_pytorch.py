import torch
from torch import nn


class VideoModelPyTorch(nn.Module):
    def __init__(self):
        super(VideoModelPyTorch, self).__init__()
        self.left_eye = nn.Sequential(
            nn.Linear(22, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 101)
        )
        self.right_eye = nn.Sequential(
            nn.Linear(22, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 101)
        )
        self.nose_mouth = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 101)
        )

    def forward(self, left_eye, right_eye, nose_mouth):
        left_eye_reshaped = left_eye.reshape(-1, 22)
        right_eye_reshaped = right_eye.reshape(-1, 22)
        nose_mouth_reshaped = nose_mouth.reshape(-1, 64)
        left_eye_params = self.left_eye(left_eye_reshaped)
        right_eye_params = self.right_eye(right_eye_reshaped)
        nose_mouth_params = self.nose_mouth(nose_mouth_reshaped)
        mat = torch.stack([left_eye_params, right_eye_params, nose_mouth_params], 0)
        result = torch.mean(mat, 0)
        return result
