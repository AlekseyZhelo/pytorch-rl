from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import torch.nn as nn

from core.model import Model
from utils.init_weights import init_weights


class A3CTargetOnlyMinisimModel(Model):
    def __init__(self, args):
        super(A3CTargetOnlyMinisimModel, self).__init__(args)

        self.lstm_layer_count = 0
        self.num_robots = args.num_robots
        self.hist_len = args.hist_len

        self.mean = np.array([0.0, 0.0, 2.30986])
        self.mean = torch.from_numpy(self.mean).float()
        self.std = np.array([0.70711, 0.70711, 1.31035])
        self.std = torch.from_numpy(self.std).float()

        # build model
        # 0. feature layers
        self.fc1 = nn.Linear(3, 16)
        self.rl1 = nn.ELU()
        self.fc2 = nn.Linear(16, 8)
        self.rl2 = nn.ELU()
        # 1. policy output
        self.policy_7 = nn.Linear(8, self.output_dims)
        self.policy_8 = nn.Softmax()
        # 2. value output
        self.value_8 = nn.Linear(8, 1)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.policy_7.bias.data.fill_(0)
        self.value_8.bias.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        target_data = x[:, :, self.input_dims[1]:self.input_dims[1] + 3 * self.num_robots * self.hist_len]
        target_data = target_data.contiguous().view(self.num_robots, 3 * self.num_robots * self.hist_len)
        laser_scans = x[:, :, :self.input_dims[1]]

        # TODO: contiguous here will slow everything down a lot?
        x = target_data

        for i in range(x.size()[0]):
            x[i, :].data -= self.mean
            x[i, :].data /= self.std

        x = self.rl1(self.fc1(x))
        x = self.rl2(self.fc2(x))
        # x = self.rl3(self.fc3(x))
        # x = x.view(-1, self.hidden_dim // 4)

        p = self.policy_7(x)
        p = self.policy_8(p)
        v = self.value_8(x)
        # if self.enable_lstm:
        # return p, v, (x, c)
        # else:
        return p, v
