from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

from core.model import Model
from utils.init_weights import init_weights, normalized_columns_initializer


class A3CMlpDeeperSeparateHiddenTwoLevelsMinisimModel(Model):
    def __init__(self, args):
        super(A3CMlpDeeperSeparateHiddenTwoLevelsMinisimModel, self).__init__(args)

        self.lstm_layer_count = 2
        self.num_robots = args.num_robots
        self.hist_len = args.hist_len
        self.hidden_vb_dim = args.hidden_vb_dim
        self.hidden_vb2_dim = args.hidden_vb_dim  # TODO: hardcoded for compatibility with previous model

        # build model
        # 0. feature layers
        self.fc1 = nn.Linear(self.input_dims[0] * self.input_dims[1], self.hidden_dim)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        self.rl3 = nn.ReLU()
        # lstm
        if self.enable_lstm:
            self.lstm = nn.LSTMCell(self.hidden_dim // 4, self.hidden_vb_dim, 1)
            self.lstm2 = nn.LSTMCell(self.hidden_vb_dim, self.hidden_vb2_dim, 1)
            final_input_size = self.hidden_vb2_dim
        else:
            final_input_size = self.hidden_dim // 4
        self.fc4 = nn.Linear(final_input_size + 2 * self.hist_len, self.hidden_dim // 2)
        self.rl4 = nn.ReLU()
        self.fc5 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)
        self.rl5 = nn.ReLU()
        # 1. policy output
        self.policy_6 = nn.Linear(self.hidden_dim // 2, self.output_dims)
        self.policy_7 = nn.Softmax()
        # 2. value output
        self.value_8 = nn.Linear(self.hidden_dim // 2, 1)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.fc1.weight.data = normalized_columns_initializer(self.fc1.weight.data, 0.01)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data = normalized_columns_initializer(self.fc2.weight.data, 0.01)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data = normalized_columns_initializer(self.fc3.weight.data, 0.01)
        self.fc3.bias.data.fill_(0)
        self.fc4.weight.data = normalized_columns_initializer(self.fc4.weight.data, 0.01)
        self.fc4.bias.data.fill_(0)
        self.fc5.weight.data = normalized_columns_initializer(self.fc5.weight.data, 0.01)
        self.fc5.bias.data.fill_(0)
        self.policy_6.weight.data = normalized_columns_initializer(self.policy_6.weight.data, 0.01)
        self.policy_6.bias.data.fill_(0)
        self.value_8.weight.data = normalized_columns_initializer(self.value_8.weight.data, 1.0)
        self.value_8.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.lstm2.bias_ih.data.fill_(0)
        self.lstm2.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None, lstm_hidden_vb2=None):
        if self.hist_len > 1:
            target_data = x[:, :, self.input_dims[1]:self.input_dims[1] + 2 * self.num_robots * self.hist_len]
            target_data = target_data.contiguous().view(target_data.size(0), 2 * self.num_robots * self.hist_len)
            laser_scans = x[:, :, :self.input_dims[1]]
        else:
            target_data = x[:, :, self.input_dims[1]:self.input_dims[1] + 2 * self.num_robots]
            target_data = target_data.contiguous().view(self.num_robots, 2)
            laser_scans = x[:, :, :self.input_dims[1]]
        # TODO: contiguous here will slow everything down a lot?
        # x = laser_scans.contiguous().view(laser_scans.size(0), self.input_dims[0] * self.input_dims[1])
        x = laser_scans.contiguous()

        x = self.rl1(self.fc1(x))
        x = self.rl2(self.fc2(x))
        x = self.rl3(self.fc3(x))
        x = x.view(-1, self.hidden_dim // 4)

        if self.enable_lstm:
            x1, c1 = self.lstm(x, lstm_hidden_vb)
            x2, c2 = self.lstm2(x1, lstm_hidden_vb2)
            x = x2

        x_aug = torch.cat((x, target_data), 1)
        x = self.rl4(self.fc4(x_aug))
        x = self.rl5(self.fc5(x))
        p = self.policy_6(x)
        p = self.policy_7(p)
        v = self.value_8(x)
        if self.enable_lstm:
            return p, v, (x1, c1), (x2, c2)
        else:
            return p, v
