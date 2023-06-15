from torch import nn


class AudioModelPyTorch(nn.Module):
    def __init__(self):
        super(AudioModelPyTorch, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=36,
                                  out_channels=36,
                                  kernel_size=5,
                                  stride=1,
                                  padding='same')
        self.batch_norm_1 = nn.BatchNorm1d(num_features=36)
        self.relu_1 = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(in_channels=36,
                                  out_channels=36,
                                  kernel_size=5,
                                  stride=1,
                                  padding='same')
        self.batch_norm_2 = nn.BatchNorm1d(num_features=36)
        self.relu_2 = nn.ReLU()
        self.lstm_1 = nn.LSTM(input_size=36,
                              hidden_size=128,
                              num_layers=1,
                              batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128,
                              hidden_size=256,
                              num_layers=1,
                              batch_first=True)
        self.lstm_3 = nn.LSTM(input_size=256,
                              hidden_size=256,
                              num_layers=1,
                              batch_first=True)
        self.linear_1 = nn.Linear(in_features=256,
                                  out_features=512)
        self.tanh_1 = nn.Tanh()
        self.linear_2 = nn.Linear(in_features=512,
                                  out_features=101)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        output = self.relu_2(self.batch_norm_2(self.conv1d_2(self.relu_1(self.batch_norm_1(self.conv1d_1(input))))))
        output = output.permute(0, 2, 1)
        output = self.lstm_3(self.lstm_2(self.lstm_1(output)[0])[0])[0]
        output = self.linear_2(self.tanh_1(self.linear_1(output)))
        return output


if __name__ == "__main__":
    pass
