from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import torch.nn as nn

from core.model import Model


class A3CCnvMinisimModel(Model):
    def __init__(self, args):
        super(A3CCnvMinisimModel, self).__init__(args)

        self.lstm_layer_count = 0
        self.num_robots = args.num_robots
        self.hist_len = args.hist_len
        self.num_filters = 8  # 16; 8

        # TODO: how-to running normalization?
        # self.register_buffer('running_mean', torch.zeros(self.input_dims[1] + 2 * self.hist_len))
        # self.register_buffer('running_var', torch.ones(self.input_dims[1] + 2 * self.hist_len))

        self.mean = np.array(
            [2.09876, 2.09931, 2.09984, 2.10028, 2.10045, 2.10068, 2.10067, 2.10062, 2.10094, 2.10138, 2.10175, 2.10205,
             2.10238, 2.10256, 2.10265, 2.10299, 2.10344, 2.10406, 2.10491, 2.10589, 2.10681, 2.10741, 2.10786, 2.10859,
             2.10953, 2.11053, 2.11154, 2.11251, 2.11323, 2.11385, 2.11462, 2.11549, 2.11645, 2.11718, 2.11749, 2.1176,
             2.11777, 2.11766, 2.11773, 2.11837, 2.1188, 2.11859, 2.1181, 2.11772, 2.1176, 2.11801, 2.11861, 2.11923,
             2.11948, 2.11946, 2.11928, 2.11924, 2.11959, 2.12011, 2.12047, 2.12056, 2.12052, 2.12039, 2.12043, 2.12068,
             2.12116, 2.12141, 2.1214, 2.12144, 2.12143, 2.12166, 2.12198, 2.12237, 2.12265, 2.12278, 2.12294,
             2.12288, 0.0, 0.0, 2.30986])
        self.mean = torch.from_numpy(self.mean).float()
        self.std = np.array(
            [1.63815, 1.63882, 1.64022, 1.64186, 1.64313, 1.64411, 1.64401, 1.64434, 1.64557, 1.64721, 1.64877, 1.64977,
             1.65044, 1.65012, 1.64996, 1.65026, 1.65162, 1.65289, 1.65421, 1.65535, 1.65579, 1.6553, 1.65472, 1.65601,
             1.65793, 1.65879, 1.6588, 1.65836, 1.65719, 1.65635, 1.65707, 1.65773, 1.6583, 1.65732, 1.65581, 1.65438,
             1.65376, 1.65345, 1.6545, 1.6566, 1.65705, 1.65489, 1.65249, 1.65084, 1.65003, 1.65158, 1.65339, 1.65523,
             1.65525, 1.65451, 1.65341, 1.65292, 1.65404, 1.65531, 1.65629, 1.65594, 1.65482, 1.65363, 1.65295, 1.65293,
             1.65371, 1.654, 1.65289, 1.6515, 1.65009, 1.64942, 1.64911, 1.64904, 1.64909, 1.64876, 1.64799, 1.6474,
             0.70711, 0.70711, 1.31035])
        self.std = torch.from_numpy(self.std).float()

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
        self.policy_7 = nn.Linear(self.hidden_dim // 2 + 3 * self.hist_len, self.output_dims)
        self.policy_8 = nn.Softmax()
        # 2. value output
        self.value_8 = nn.Linear(self.hidden_dim // 2 + 3 * self.hist_len, 1)

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
        # shape = x.size()
        # x = self.bn0(x.view(self.num_robots, -1))
        # x = x.view(shape)
        for i in range(x.size()[1]):
            x[0, i, :].data -= self.mean
            x[0, i, :].data /= self.std

        target_data = x[:, :, self.input_dims[1]:self.input_dims[1] + 3 * self.num_robots * self.hist_len]
        target_data = target_data.contiguous().view(self.num_robots, 3 * self.num_robots * self.hist_len)
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
