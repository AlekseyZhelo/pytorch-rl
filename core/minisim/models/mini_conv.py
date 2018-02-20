from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import torch.nn as nn

from core.minisim.util.normalization import get_state_statistics, apply_normalization
from core.model import Model


class A3CCnvMinisimModel(Model):
    def __init__(self, args):
        super(A3CCnvMinisimModel, self).__init__(args)

        self.lstm_layer_count = 0
        self.num_robots = args.num_robots
        self.hist_len = args.hist_len
        self.target_data_dim = args.target_data_dim
        self.num_filters = 8  # 16; 4

        self.mean, self.std = get_state_statistics()

        # build model
        # 0. feature layers
        # self.bn0 = nn.BatchNorm1d(self.input_dims[1] + 2 * self.hist_len)
        self.cnv1 = nn.Conv1d(1, self.num_filters, 5, 2)
        self.sz_1 = (self.input_dims[0] * self.input_dims[1] - 5) // 2 + 1
        self.rl1 = nn.ELU()
        self.cnv2 = nn.Conv1d(self.num_filters, self.num_filters, 3, 2)
        self.sz_2 = (self.sz_1 - 3) // 2 + 1
        self.rl2 = nn.ELU()
        self.fc3 = nn.Linear(self.sz_2 * self.num_filters, self.hidden_dim)
        self.rl3 = nn.ELU()
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.rl4 = nn.ELU()

        # 1. policy output
        self.policy_7 = nn.Linear(self.hidden_dim // 2 + self.target_data_dim * self.hist_len, self.output_dims)
        self.policy_8 = nn.Softmax()
        # 2. value output
        self.value_8 = nn.Linear(self.hidden_dim // 2 + self.target_data_dim * self.hist_len, 1)

        self._reset()

    def _init_weights(self):
        nn.init.xavier_uniform(self.cnv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.cnv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.fc3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.fc4.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.policy_7.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.value_8.weight.data, gain=nn.init.calculate_gain('relu'))
        self.cnv1.bias.data.fill_(0)
        self.cnv2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        self.fc4.bias.data.fill_(0)
        self.policy_7.bias.data.fill_(0)
        self.value_8.bias.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        apply_normalization(x, self.mean, self.std)

        target_data = x[:, :, self.input_dims[1]:self.input_dims[1]
                                                 + self.target_data_dim * self.hist_len]
        target_data = target_data.contiguous().view(self.num_robots,
                                                    self.target_data_dim * self.hist_len)
        laser_scans = x[:, :, :self.input_dims[1]]

        # TODO: contiguous here will slow everything down a lot?
        x = laser_scans.contiguous()
        x = x.view(self.num_robots, 1, self.input_dims[1])

        x = self.rl1(self.cnv1(x))
        x = self.rl2(self.cnv2(x))
        x = x.view(self.num_robots, self.sz_2 * self.num_filters)
        x = self.rl3(self.fc3(x))
        x = self.rl4(self.fc4(x))

        x_aug = torch.cat((x, target_data), 1)
        p = self.policy_7(x_aug)
        p = self.policy_8(p)
        v = self.value_8(x_aug)
        return p, v
