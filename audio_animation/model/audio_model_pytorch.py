from torch import nn


class AudioModelPyTorch(nn.Module):
    def __init__(self):
        super(AudioModelPyTorch, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=29,
                      out_channels=32,
                      kernel_size=(5, 1),
                      stride=(1, 1)),
            nn.ReLU(),
            nn.Conv1d(in_channels=32,
                      out_channels=32,
                      kernel_size=(5, 1),
                      stride=(1, 1)),
            nn.ReLU(),
            nn.LSTM(input_size=32,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True),
            nn.LSTM(input_size=128,
                    hidden_size=64,
                    num_layers=2,
                    batch_first=True),
            nn.Linear(in_features=64,
                      out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128,
                      out_features=101)
        )

    def forward(self, input_data, h_0, c_0):
        output_data = self.model(input_data, (h_0, c_0))
        return output_data


if __name__ == "__main__":
    pass
