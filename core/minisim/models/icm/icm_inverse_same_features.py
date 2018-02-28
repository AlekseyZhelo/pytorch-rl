from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

from core.model import Model
from utils.init_weights import init_weights, normalized_columns_initializer


class ICMInverseModelSameFeatures(Model):
    def __init__(self, args):
        super(ICMInverseModelSameFeatures, self).__init__(args)

        self.lstm_layer_count = 0
        self.num_robots = args.num_robots

        self.hidden_dim = args.icm_inv_hidden_dim
        self.hidden_vb_dim = args.icm_inv_hidden_vb_dim
        self.feature_dim = args.icm_feature_dim
        self.output_dims = args.action_dim  # same as the superclass, repeated for clarity

        # build model
        # 0. feature layers
        self.fc1 = nn.Linear(self.feature_dim, self.hidden_dim)
        self.rl1 = nn.ELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.rl2 = nn.ELU()
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.feature_dim)
        self.rl3 = nn.Tanh()

        self.fc4 = nn.Linear(2 * self.feature_dim, 2 * self.feature_dim)
        self.rl4 = nn.ELU()
        # 1. action output
        self.action_5 = nn.Linear(2 * self.feature_dim, self.output_dims)
        self.action_6 = nn.Softmax()

        self._reset()

    def _init_weights(self):
        nn.init.xavier_uniform(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.fc2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.fc3.weight.data, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform(self.fc4.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.action_5.weight.data, gain=nn.init.calculate_gain('linear'))
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        self.fc4.bias.data.fill_(0)
        self.action_5.bias.data.fill_(0)

    def forward(self, input_):
        a3c_features, a3c_features_next = input_

        x_1 = self.rl1(self.fc1(a3c_features))
        x_1 = self.rl2(self.fc2(x_1))
        x_1 = self.rl3(self.fc3(x_1))

        x_2 = self.rl1(self.fc1(a3c_features_next))
        x_2 = self.rl2(self.fc2(x_2))
        x_2 = self.rl3(self.fc3(x_2))

        x = torch.cat(
            (x_1, x_2),
            1
        )

        x = self.rl4(self.fc4(x))

        logits = self.action_5(x)
        probs = self.action_6(logits)

        return x_1, x_2, logits, probs

    @staticmethod
    def same_features():
        return True
