from torch import nn
import torch


class AudioModelPyTorch(nn.Module):
    def __init__(self):
        super(AudioModelPyTorch, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=29,
                                  out_channels=32,
                                  kernel_size=5,
                                  stride=1,
                                  padding='same')
        self.relu_1 = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(in_channels=32,
                                  out_channels=32,
                                  kernel_size=5,
                                  stride=1,
                                  padding='same')
        self.relu_2 = nn.ReLU()
        self.lstm_1 = nn.LSTM(input_size=32,
                              hidden_size=128,
                              num_layers=2,
                              batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128,
                              hidden_size=64,
                              num_layers=2,
                              batch_first=True)
        self.linear_1 = nn.Linear(in_features=64,
                                  out_features=128)
        self.tanh_1 = nn.Tanh()
        self.linear_2 = nn.Linear(in_features=128,
                                  out_features=101)
        # self.model = nn.Sequential(
        #     nn.Conv1d(in_channels=29,
        #               out_channels=32,
        #               kernel_size=(5, 1),
        #               stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=32,
        #               out_channels=32,
        #               kernel_size=(5, 1),
        #               stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.LSTM(input_size=32,
        #             hidden_size=128,
        #             num_layers=2,
        #             batch_first=True),
        #     nn.LSTM(input_size=128,
        #             hidden_size=64,
        #             num_layers=2,
        #             batch_first=True),
        #     nn.Linear(in_features=64,
        #               out_features=128),
        #     nn.Tanh(),
        #     nn.Linear(in_features=128,
        #               out_features=101)
        # )

    def forward(self, input):
        input = input.permute(0, 2, 1)
        # output = self.conv1d_1(input)
        # output = self.relu_1(output)
        # output = self.conv1d_2(output)
        # output = self.relu_2(output)
        output = self.relu_2(self.conv1d_2(self.relu_1(self.conv1d_1(input))))
        output = output.permute(0, 2, 1)
        output = self.lstm_2(self.lstm_1(output)[0])[0]
        # output, _ = self.lstm_2(output)
        output = self.linear_2(self.tanh_1(self.linear_1(output)))
        return output


if __name__ == "__main__":
    # m = nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3, stride=2)
    # input = torch.randn(20, 16, 50)
    # output = m(input)
    pass
