import torch
import torch.nn as nn


class IKModel(nn.Module):
    def __init__(self):
        super(IKModel, self).__init__()
        self.left_eye = nn.Sequential(
            nn.Linear(33, 256),
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
            nn.Linear(256, 136)
        )
        self.right_eye = nn.Sequential(
            nn.Linear(33, 256),
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
            nn.Linear(256, 136)
        )
        self.nose_mouth = nn.Sequential(
            nn.Linear(87, 256),
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
            nn.Linear(256, 136)
        )

    def forward(self, left_eye, right_eye, nose_mouth):
        le_reshaped = left_eye.reshape(-1, 33)
        re_reshaped = right_eye.reshape(-1, 33)
        nm_reshaped = nose_mouth.reshape(-1, 87)
        le_params = self.left_eye(le_reshaped)
        re_params = self.right_eye(re_reshaped)
        nm_params = self.nose_mouth(nm_reshaped)
        mat = torch.stack([le_params, re_params, nm_params], 0)
        result = torch.mean(mat, 0)
        return result


if __name__ == '__main__':
    pass
