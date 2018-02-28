from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.optim as optim

from utils.helpers import Experience

class Agent(object):
    def __init__(self, args, env_prototype, model_prototype, memory_prototype=None):
        # logging
        self.logger = args.logger

        # prototypes for env & model & memory
        self.env_prototype = env_prototype          # NOTE: instantiated in fit_model() of inherited Agents
        self.env_params = args.env_params
        self.model_prototype = model_prototype      # NOTE: instantiated in fit_model() of inherited Agents
        self.model_params = args.model_params
        self.memory_prototype = memory_prototype    # NOTE: instantiated in __init__()  of inherited Agents (dqn needs, a3c doesn't so only pass in None)
        self.memory_params = args.memory_params

        # params
        self.model_name = args.model_name           # NOTE: will save the current model to model_name
        self.model_file = args.model_file           # NOTE: will load pretrained model_file if not None

        # icm
        self.icm = args.icm if hasattr(args, "icm") else False
        if self.icm:
            self.icm_inv_lr = args.icm_inv_lr
            self.icm_fwd_lr = args.icm_fwd_lr
            self.icm_inv_model_prototype = args.icm_inv_model
            self.icm_fwd_model_prototype = args.icm_fwd_model
            self.icm_inv_model = None  # initialize in subclasses
            self.icm_fwd_model = None  # initialize in subclasses

            self.icm_inv_model_name = args.icm_inv_model_name
            self.icm_inv_model_file = args.icm_inv_model_file
            self.icm_fwd_model_name = args.icm_fwd_model_name
            self.icm_fwd_model_file = args.icm_fwd_model_file

        self.render = args.render
        self.visualize = args.visualize
        if self.visualize:
            self.vis = args.vis
            self.refs = args.refs

        self.save_best = args.save_best
        self.icm_save_best = args.icm_save_best
        if self.save_best:
            self.best_step   = None                 # NOTE: achieves best_reward at this step
            self.best_reward = None                 # NOTE: only save a new model if achieves higher reward
        if self.icm_save_best:
            self.best_icm_inv_step = None
            self.best_icm_inv_loss = None
            self.best_icm_fwd_step = None
            self.best_icm_fwd_loss = None

        self.hist_len = args.hist_len
        self.hidden_dim = args.model_params.hidden_dim

        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        # agent_params
        # criteria and optimizer
        self.value_criteria = args.value_criteria
        self.optim = args.optim
        # hyperparameters
        self.steps = args.steps
        self.early_stop = args.early_stop
        self.gamma = args.gamma
        self.clip_grad = args.clip_grad
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.eval_freq = args.eval_freq
        self.eval_steps = args.eval_steps
        self.prog_freq = args.prog_freq
        self.test_nepisodes = args.test_nepisodes
        if args.agent_type == "dqn":
            self.enable_double_dqn  = args.enable_double_dqn
            self.enable_dueling = args.enable_dueling
            self.dueling_type = args.dueling_type

            self.learn_start = args.learn_start
            self.batch_size = args.batch_size
            self.valid_size = args.valid_size
            self.eps_start = args.eps_start
            self.eps_end = args.eps_end
            self.eps_eval = args.eps_eval
            self.eps_decay = args.eps_decay
            self.target_model_update = args.target_model_update
            self.action_repetition = args.action_repetition
            self.memory_interval = args.memory_interval
            self.train_interval = args.train_interval
        elif args.agent_type == "a3c":
            self.enable_log_at_train_step = args.enable_log_at_train_step

            self.enable_lstm = args.enable_lstm
            self.enable_continuous = args.enable_continuous
            self.num_processes = args.num_processes

            self.rollout_steps = args.rollout_steps
            self.tau = args.tau
            self.beta = args.beta
            self.icm_beta = args.icm_beta
            self.icm_fwd_wt = args.icm_fwd_wt
        elif args.agent_type == "acer":
            self.enable_bias_correction = args.enable_bias_correction
            self.enable_1st_order_trpo = args.enable_1st_order_trpo
            self.enable_log_at_train_step = args.enable_log_at_train_step

            self.enable_lstm = args.enable_lstm
            self.enable_continuous = args.enable_continuous
            self.num_processes = args.num_processes

            self.replay_ratio = args.replay_ratio
            self.replay_start = args.replay_start
            self.batch_size = args.batch_size
            self.valid_size = args.valid_size
            self.clip_trace = args.clip_trace
            self.clip_1st_order_trpo = args.clip_1st_order_trpo
            self.avg_model_decay = args.avg_model_decay

            self.rollout_steps = args.rollout_steps
            self.tau = args.tau
            self.beta = args.beta

    def _reset_experience(self):
        self.experience = Experience(state0 = None,
                                     action = None,
                                     reward = None,
                                     state1 = None,
                                     terminal1 = False)

    def _load_model(self, model_file):
        if model_file:
            self.logger.warning("Loading Model: " + self.model_file + " ...")
            self.model.load_state_dict(torch.load(model_file))
            self.logger.warning("Loaded  Model: " + self.model_file + " ...")
        else:
            self.logger.warning("No Pretrained Model. Will Train From Scratch.")

    def _save_model(self, step, curr_reward):
        self.logger.warning("Saving Model    @ Step: " + str(step) + ": " + self.model_name + " ...")
        if self.save_best:
            if self.best_step is None:
                self.best_step   = step
                self.best_reward = curr_reward
            if curr_reward >= self.best_reward:
                self.best_step   = step
                self.best_reward = curr_reward
                torch.save(self.model.state_dict(), self.model_name)
            self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.model_name + ". {Best Step: " + str(self.best_step) + " | Best Reward: " + str(self.best_reward) + "}")
        else:
            idx = self.model_name.index('.pth')
            step_name = self.model_name[:idx] + '_step_' + str(step) + '.pth'
            torch.save(self.model.state_dict(), step_name)
            self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.model_name + ".")

    # TODO: used everywhere where needed? (a3c.py)
    def _load_icm_models(self, inv_model_file, fwd_model_file):
        if inv_model_file and fwd_model_file:
            self.logger.warning("Loading ICM Inverse Model: " + inv_model_file + " ...")
            self.icm_inv_model.load_state_dict(torch.load(inv_model_file))
            self.logger.warning("Loaded ICM Inverse  Model: " + inv_model_file + " ...")
            self.logger.warning("Loading ICM Forward Model: " + fwd_model_file + " ...")
            self.icm_fwd_model.load_state_dict(torch.load(fwd_model_file))
            self.logger.warning("Loaded ICM Forward  Model: " + fwd_model_file + " ...")
        else:
            self.logger.warning("No Pretrained ICM Models. Will Train From Scratch.")

    # TODO: used everywhere where needed? (a3c_single_process.py)
    def _save_icm_models(self, step, curr_inv_loss, curr_fwd_loss):
        self.logger.warning("Saving ICM Inverse Model    @ Step: " + str(step) + ": "
                            + self.icm_inv_model_name + " ...")
        if self.icm_save_best:
            if self.best_icm_inv_step is None:
                self.best_icm_inv_step = step
                self.best_icm_inv_loss = curr_inv_loss
            if curr_inv_loss < self.best_icm_inv_loss:
                self.best_icm_inv_step = step
                self.best_icm_inv_loss = curr_inv_loss
                torch.save(self.icm_inv_model.state_dict(), self.icm_inv_model_name)
            self.logger.warning(
                "Saved ICM Inverse Model    @ Step: " + str(step) + ": " + self.icm_inv_model_name + ". {Best Step: " + str(
                    self.best_icm_inv_step) + " | Best Loss: " + str(self.best_icm_inv_loss) + "}")

            self.logger.warning("Saving ICM Forward Model    @ Step: " + str(step) + ": "
                                + self.icm_fwd_model_name + " ...")
            if self.best_icm_fwd_step is None:
                self.best_icm_fwd_step = step
                self.best_icm_fwd_loss = curr_fwd_loss
            if curr_fwd_loss < self.best_icm_fwd_loss:
                self.best_icm_fwd_step = step
                self.best_icm_fwd_loss = curr_fwd_loss
                torch.save(self.icm_fwd_model.state_dict(), self.icm_fwd_model_name)
            self.logger.warning(
                "Saved ICM Forward Model    @ Step: " + str(step) + ": " + self.icm_fwd_model_name + ". {Best Step: " + str(
                    self.best_icm_fwd_step) + " | Best Loss: " + str(self.best_icm_fwd_loss) + "}")
        else:
            idx = self.icm_inv_model_name.index('.pth')
            inv_step_name = self.icm_inv_model_name[:idx] + '_step_' + str(step) + '.pth'
            torch.save(self.icm_inv_model.state_dict(), inv_step_name)
            self.logger.warning("Saved ICM Inverse Model    @ Step: " + str(step) + ": " + self.icm_inv_model_name + ".")

            self.logger.warning("Saving ICM Forward Model    @ Step: " + str(step) + ": " + self.icm_fwd_model_name + " ...")
            idx = self.icm_fwd_model_name.index('.pth')
            fwd_step_name = self.icm_fwd_model_name[:idx] + '_step_' + str(step) + '.pth'
            torch.save(self.icm_fwd_model.state_dict(), fwd_step_name)
            self.logger.warning("Saved ICM Forward  Model    @ Step: " + str(step) + ": " + self.icm_fwd_model_name + ".")

    def _forward(self, observation):
        raise NotImplementedError("not implemented in base calss")

    def _backward(self, reward, terminal):
        raise NotImplementedError("not implemented in base calss")

    def _eval_model(self):  # evaluation during training
        raise NotImplementedError("not implemented in base calss")

    def fit_model(self):    # training
        raise NotImplementedError("not implemented in base calss")

    def test_model(self):   # testing pre-trained models
        raise NotImplementedError("not implemented in base calss")
