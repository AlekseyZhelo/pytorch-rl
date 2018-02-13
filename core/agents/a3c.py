from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.multiprocessing as mp

from core.agent import Agent
from core.agents.a3c_single_process import A3CLearner, A3CEvaluator, A3CTester


class A3CAgent(Agent):
    def __init__(self, args, env_prototype, model_prototype, memory_prototype):
        super(A3CAgent, self).__init__(args, env_prototype, model_prototype, memory_prototype)
        self.logger.warning("<===================================> A3C-Master {Env(dummy) & Model}")

        if hasattr(args, "num_robots"):
            self.num_robots = args.num_robots

        # dummy_env just to get state_shape & action_dim
        self.dummy_env   = self.env_prototype(self.env_params, self.num_processes + 1)
        self.state_shape = self.dummy_env.state_shape
        self.action_dim  = self.dummy_env.action_dim
        del self.dummy_env

        self.logger.warning("<===================================> A3C-Master {After dummy env}")
        # global shared model
        self.model_params.state_shape = self.state_shape
        self.model_params.action_dim  = self.action_dim
        self.model = self.model_prototype(self.model_params)
        self._load_model(self.model_file)   # load pretrained model if provided
        self.model.share_memory()           # NOTE

        if hasattr(self.model, "lstm_layer_count"):
            if self.model.lstm_layer_count == 0:
                self.enable_lstm = False

        if self.icm:
            self.icm_inv_model = self.icm_inv_model_prototype(self.model_params)
            self.icm_fwd_model = self.icm_fwd_model_prototype(self.model_params)
            self._load_icm_models(self.icm_inv_model_file, self.icm_fwd_model_file)
            self.icm_inv_model.share_memory()
            self.icm_fwd_model.share_memory()

        # learning algorithm
        self.optimizer    = self.optim(self.model.parameters(), lr = self.lr)
        self.optimizer.share_memory()       # NOTE
        self.lr_adjusted  = mp.Value('d', self.lr) # adjusted lr
        if self.icm:
            self.icm_inv_optimizer = self.optim(self.icm_inv_model.parameters(), lr=self.icm_inv_lr)
            self.icm_inv_optimizer.share_memory()
            self.icm_inv_lr_adjusted = mp.Value('d', self.icm_inv_lr)  # adjusted lr

            self.icm_fwd_optimizer = self.optim(self.icm_fwd_model.parameters(), lr=self.icm_fwd_lr)
            self.icm_fwd_optimizer.share_memory()
            self.icm_fwd_lr_adjusted = mp.Value('d', self.icm_fwd_lr)  # adjusted lr

        # global counters
        self.frame_step   = mp.Value('l', 0) # global frame step counter
        self.train_step   = mp.Value('l', 0) # global train step counter
        self.terminations_count   = mp.Value('l', 0) # global train step counter
        # global training stats
        self.p_loss_avg   = mp.Value('d', 0.) # global policy loss
        self.v_loss_avg   = mp.Value('d', 0.) # global value loss
        self.loss_avg     = mp.Value('d', 0.) # global loss
        self.loss_counter = mp.Value('l', 0)  # storing this many losses
        self.grad_magnitude_avg = mp.Value('d', 0.)
        self.grad_magnitude_max = mp.Value('d', 0.)
        if self.icm:
            self.icm_inv_loss_avg = mp.Value('d', 0.)  # global ICM inverse loss
            self.icm_fwd_loss_avg = mp.Value('d', 0.)  # global ICM forward loss
            self.icm_inv_accuracy_avg = mp.Value('d', 0.)
        self._reset_training_logs()

    def _reset_training_logs(self):
        self.p_loss_avg.value   = 0.
        self.v_loss_avg.value   = 0.
        self.loss_avg.value     = 0.
        self.loss_counter.value = 0
        self.grad_magnitude_avg.value = 0.
        self.grad_magnitude_max.value = 0.
        if self.icm:
            self.icm_inv_loss_avg.value = 0.
            self.icm_fwd_loss_avg.value = 0.
            self.icm_inv_accuracy_avg.value = 0.

    def fit_model(self):
        self.jobs = []
        for process_id in range(self.num_processes):
            self.jobs.append(A3CLearner(self, process_id))
        self.jobs.append(A3CEvaluator(self, self.num_processes))  # TODO: uncomment when finished
        # print ("Warning: NOT STARTING THE EVALUATOR")
        # self.logger.warning("<===================================> Warning: NOT STARTING THE EVALUATOR ...")
        # TODO: figure out why the logs stop printing to console

        self.logger.warning("<===================================> Training ...")
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()

    def test_model(self):
        self.jobs = []
        self.jobs.append(A3CTester(self))  # TODO: add small chance for random actions to prevent stalling in place
        # TODO: check performance on a similar, but slightly changed map

        self.logger.warning("<===================================> Testing ...")
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()
