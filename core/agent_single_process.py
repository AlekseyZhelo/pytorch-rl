from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import time
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils.helpers import Experience, one_hot


class AgentSingleProcess(mp.Process):
    def __init__(self, master, process_id=0):
        super(AgentSingleProcess, self).__init__(name = "Process-%d" % process_id)
        # NOTE: self.master.* refers to parameters shared across all processes
        # NOTE: self.*        refers to process-specific properties
        # NOTE: we are not copying self.master.* to self.* to keep the code clean

        self.master = master
        self.process_id = process_id

        # env
        self.env = self.master.env_prototype(self.master.env_params, self.process_id)
        # model
        self.model = self.master.model_prototype(self.master.model_params)
        if self.master.icm:
            self.icm_inv_model = self.master.icm_inv_model_prototype(self.master.model_params)
            self.icm_fwd_model = self.master.icm_fwd_model_prototype(self.master.model_params)
            self.icm_inv_loss_criterion = torch.nn.CrossEntropyLoss(reduce=False)
            self.icm_fwd_loss_criterion = torch.nn.MSELoss(reduce=False)
        self._sync_local_with_global()

        # experience
        self._reset_experience()

    def _reset_experience(self):    # for getting one set of observation from env for every action taken
        self.experience = Experience(state0 = None,
                                     action = None,
                                     reward = None,
                                     state1 = None,
                                     terminal1 = False)

    def _sync_local_with_global(self):  # grab the current global model for local learning/evaluating
        self.model.load_state_dict(self.master.model.state_dict())
        if self.master.icm:
            self.icm_inv_model.load_state_dict(self.master.icm_inv_model.state_dict())
            self.icm_fwd_model.load_state_dict(self.master.icm_fwd_model.state_dict())

    # NOTE: since no backward passes has ever been run on the global model
    # NOTE: its grad has never been initialized, here we ensure proper initialization
    # NOTE: reference: https://discuss.pytorch.org/t/problem-on-variable-grad-data/957
    def _ensure_global_grads(self):
        for global_param, local_param in zip(self.master.model.parameters(),
                                             self.model.parameters()):
            if global_param.grad is not None:
                # print("proc_id: {1}, same var: {0}".format(local_param.grad is global_param._grad,
                #                                             self.process_id))
                # print(global_param._grad.__repr__())
                break
            global_param._grad = local_param.grad

        if self.master.icm:  # TODO: will work the same?
            for global_param, local_param in zip(self.master.icm_inv_model.parameters(),
                                                 self.icm_inv_model.parameters()):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad

            for global_param, local_param in zip(self.master.icm_fwd_model.parameters(),
                                                 self.icm_fwd_model.parameters()):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad

    def _forward(self, observation):
        raise NotImplementedError("not implemented in base class")

    def _backward(self, reward, terminal):
        raise NotImplementedError("not implemented in base class")

    def run(self):
        raise NotImplementedError("not implemented in base class")
