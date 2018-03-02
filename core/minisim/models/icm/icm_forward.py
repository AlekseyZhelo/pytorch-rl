from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

from core.model import Model
from utils import helpers
from utils.init_weights import init_weights, normalized_columns_initializer


class ICMForwardModel(Model):
    def __init__(self, args):
        super(ICMForwardModel, self).__init__(args)

        self.lstm_layer_count = 0
        self.num_robots = args.num_robots

        self.hidden_dim = args.icm_fwd_hidden_dim
        self.hidden_vb_dim = args.icm_fwd_hidden_vb_dim
        self.feature_dim = args.icm_feature_dim
        self.action_dim = args.action_dim  # same as the superclass, repeated for clarity

        # build model
        # 0. next feature prediction
        self.fc1 = nn.Linear(self.feature_dim + self.action_dim, self.hidden_dim)
        self.rl1 = nn.ELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.rl2 = nn.ELU()
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.feature_dim)
        # self.rl3 = nn.Tanh()  # no activation here for regression!
        # keep tanh for intrinsic reward calculation, train forward on raw before tanh

        self._reset()

    def _init_weights(self):
        nn.init.xavier_uniform(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.fc2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.fc3.weight.data, gain=nn.init.calculate_gain('linear'))
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)

    def forward(self, input_):
        state_features, action = input_

        x = torch.cat(
            (state_features,
             Variable(torch.from_numpy(helpers.one_hot(self.action_dim, action.data.numpy())).type(self.dtype))),
            1
        )

        x = self.rl1(self.fc1(x))
        x = self.rl2(self.fc2(x))
        # x = self.rl3(self.fc3(x))
        x = self.fc3(x)

        return x
