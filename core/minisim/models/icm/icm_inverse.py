from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

from core.model import Model
from utils.init_weights import init_weights, normalized_columns_initializer


# TODO: try LSTM?
class ICMInverseModel(Model):
    def __init__(self, args):
        super(ICMInverseModel, self).__init__(args)

        self.lstm_layer_count = 0
        self.num_robots = args.num_robots

        self.hidden_dim = args.icm_inv_hidden_dim
        self.hidden_vb_dim = args.icm_inv_hidden_vb_dim
        self.feature_dim = args.icm_feature_dim
        self.output_dims = args.action_dim  # same as the superclass, repeated for clarity

        # build model
        # 0. feature layers
        self.fc1 = nn.Linear(self.input_dims[1], self.hidden_dim)
        self.rl1 = nn.ELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.rl2 = nn.ELU()
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.feature_dim)
        self.rl3 = nn.ELU()

        # 1. action output  # TODO: more layers here?
        self.action_4 = nn.Linear(2 * self.feature_dim, self.output_dims)
        self.action_5 = nn.Softmax()

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.fc1.weight.data = normalized_columns_initializer(self.fc1.weight.data, 0.01)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data = normalized_columns_initializer(self.fc2.weight.data, 0.01)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data = normalized_columns_initializer(self.fc3.weight.data, 0.01)
        self.fc3.bias.data.fill_(0)
        self.action_4.weight.data = normalized_columns_initializer(self.action_4.weight.data, 0.01)
        self.action_4.bias.data.fill_(0)

    def forward(self, input_):
        state, state_next = input_

        x_1 = self.rl1(self.fc1(state))
        x_1 = self.rl2(self.fc2(x_1))
        x_1 = self.rl3(self.fc3(x_1))

        x_2 = self.rl1(self.fc1(state_next))
        x_2 = self.rl2(self.fc2(x_2))
        x_2 = self.rl3(self.fc3(x_2))

        x = torch.cat(
            (x_1, x_2),
            1
        )

        p_lin = self.action_4(x)
        p = self.action_5(p_lin)

        return x_1, x_2, p_lin, p
