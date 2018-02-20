from __future__ import absolute_import
from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.minisim.util.normalization import apply_normalization, get_state_statistics
from core.model import Model
from utils.init_weights import init_weights, normalized_columns_initializer


class A3CMlpNarrowingMinisimModel(Model):
    def __init__(self, args):
        super(A3CMlpNarrowingMinisimModel, self).__init__(args)

        self.lstm_layer_count = 1
        self.num_robots = args.num_robots
        self.hist_len = args.hist_len
        self.target_data_dim = args.target_data_dim
        self.hidden_vb_dim = args.hidden_vb_dim

        self.mean, self.std = get_state_statistics()

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
            final_input_size = self.hidden_vb_dim
        else:
            final_input_size = self.hidden_dim // 4
        # 1. policy output
        self.policy_5 = nn.Linear(final_input_size + self.target_data_dim * self.hist_len, self.output_dims)
        self.policy_6 = nn.Softmax()
        # 2. value output
        self.value_5 = nn.Linear(final_input_size + self.target_data_dim * self.hist_len, 1)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        nn.init.xavier_normal(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.fc2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.fc3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.lstm.weight_hh.data, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_normal(self.lstm.weight_ih.data, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_normal(self.policy_5.weight.data, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_normal(self.value_5.weight.data, gain=nn.init.calculate_gain('linear'))

        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.policy_5.bias.data.fill_(0)
        self.value_5.bias.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        apply_normalization(x, self.mean, self.std)

        target_data = x[:, :, self.input_dims[1]:self.input_dims[1]
                                                 + self.target_data_dim * self.num_robots * self.hist_len]
        target_data = target_data.contiguous().view(self.num_robots,
                                                    self.target_data_dim * self.num_robots * self.hist_len)
        laser_scans = x[:, :, :self.input_dims[1]]

        # TODO: contiguous here will slow everything down a lot?
        x = laser_scans.contiguous()

        x = self.rl1(self.fc1(x))
        x = self.rl2(self.fc2(x))
        x = self.rl3(self.fc3(x))
        x = x.view(-1, self.hidden_dim // 4)

        if self.enable_lstm:
            x, c = self.lstm(x, lstm_hidden_vb)

        x_aug = torch.cat((x, target_data), 1)
        p = self.policy_5(x_aug)
        p = self.policy_6(p)
        v = self.value_5(x_aug)
        if self.enable_lstm:
            return p, v, (x, c)
        else:
            return p, v
