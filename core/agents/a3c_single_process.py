from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D

from core.agent_single_process import AgentSingleProcess
from utils.helpers import A3C_Experience
from utils.legend_handlers import HandlerCircle, HandlerRectangle


class A3CSingleProcess(AgentSingleProcess):
    epsilon = 1e-13

    def __init__(self, master, process_id=0):
        super(A3CSingleProcess, self).__init__(master, process_id)

        # lstm hidden states
        if self.master.enable_lstm:
            self.lstm_layer_count = self.model.lstm_layer_count if hasattr(self.model, "lstm_layer_count") else 1
            self._reset_lstm_hidden_vb_episode()  # clear up hidden state
            self._reset_lstm_hidden_vb_rollout()  # detach the previous variable from the computation graph

        # NOTE global variable pi
        if self.master.enable_continuous:
            self.pi_vb = Variable(torch.Tensor([math.pi]).type(self.master.dtype))

        self.master.logger.warning(
            "Registered A3C-SingleProcess-Agent #" + str(self.process_id) + " w/ Env (seed:" + str(
                self.env.seed) + ").")

    # NOTE: to be called at the beginning of each new episode, clear up the hidden state
    # TODO: also reset for ICM if it uses LSTM!
    def _reset_lstm_hidden_vb_episode(self, training=True):  # seq_len, batch_size, hidden_vb_dim
        not_training = not training

        if hasattr(self.master, "num_robots"):
            r = self.master.num_robots
            if self.master.enable_continuous:  # TODO: what here?
                self.lstm_hidden_vb = (
                    Variable(torch.zeros(2, self.model.hidden_vb_dim).type(self.master.dtype), volatile=not_training),
                    Variable(torch.zeros(2, self.model.hidden_vb_dim).type(self.master.dtype), volatile=not_training))
                if self.lstm_layer_count == 2:
                    self.lstm_hidden_vb2 = (
                        Variable(torch.zeros(2, self.model.hidden_vb2_dim).type(self.master.dtype),
                                 volatile=not_training),
                        Variable(torch.zeros(2, self.model.hidden_vb2_dim).type(self.master.dtype),
                                 volatile=not_training))
            else:
                self.lstm_hidden_vb = (
                    Variable(torch.zeros(r, self.model.hidden_vb_dim).type(self.master.dtype), volatile=not_training),
                    Variable(torch.zeros(r, self.model.hidden_vb_dim).type(self.master.dtype), volatile=not_training))
                if self.lstm_layer_count == 2:
                    self.lstm_hidden_vb2 = (
                        Variable(torch.zeros(r, self.model.hidden_vb2_dim).type(self.master.dtype),
                                 volatile=not_training),
                        Variable(torch.zeros(r, self.model.hidden_vb2_dim).type(self.master.dtype),
                                 volatile=not_training))
        else:
            if self.master.enable_continuous:
                self.lstm_hidden_vb = (
                    Variable(torch.zeros(2, self.model.hidden_vb_dim).type(self.master.dtype), volatile=not_training),
                    Variable(torch.zeros(2, self.model.hidden_vb_dim).type(self.master.dtype), volatile=not_training))
                if self.lstm_layer_count == 2:
                    self.lstm_hidden_vb2 = (
                        Variable(torch.zeros(2, self.model.hidden_vb2_dim).type(self.master.dtype),
                                 volatile=not_training),
                        Variable(torch.zeros(2, self.model.hidden_vb2_dim).type(self.master.dtype),
                                 volatile=not_training))
            else:
                self.lstm_hidden_vb = (
                    Variable(torch.zeros(1, self.model.hidden_vb_dim).type(self.master.dtype), volatile=not_training),
                    Variable(torch.zeros(1, self.model.hidden_vb_dim).type(self.master.dtype), volatile=not_training))
                if self.lstm_layer_count == 2:
                    self.lstm_hidden_vb2 = (
                        Variable(torch.zeros(1, self.model.hidden_vb2_dim).type(self.master.dtype),
                                 volatile=not_training),
                        Variable(torch.zeros(1, self.model.hidden_vb2_dim).type(self.master.dtype),
                                 volatile=not_training))

    # NOTE: to be called at the beginning of each rollout, detach the previous variable from the graph
    def _reset_lstm_hidden_vb_rollout(self):
        self.lstm_hidden_vb = (Variable(self.lstm_hidden_vb[0].data),
                               Variable(self.lstm_hidden_vb[1].data))
        if self.lstm_layer_count == 2:
            self.lstm_hidden_vb2 = (Variable(self.lstm_hidden_vb2[0].data),
                                    Variable(self.lstm_hidden_vb2[1].data))

    def _preprocessState(self, state, is_valotile=False):
        if isinstance(state, list):
            state_vb = []
            for i in range(len(state)):
                state_vb.append(
                    Variable(torch.from_numpy(state[i]).unsqueeze(0).type(self.master.dtype), volatile=is_valotile))
        else:
            state_vb = Variable(torch.from_numpy(state).unsqueeze(0).type(self.master.dtype), volatile=is_valotile)
        return state_vb

    def _forward(self, state_vb, off_record=False):
        if self.master.enable_continuous:  # NOTE continuous control p_vb here is the mu_vb of continuous action dist
            if self.master.enable_lstm:
                if self.lstm_layer_count == 1:
                    if off_record:
                        p_vb, sig_vb, v_vb, _ = self.model(state_vb, self.lstm_hidden_vb)
                    else:
                        p_vb, sig_vb, v_vb, self.lstm_hidden_vb = self.model(state_vb, self.lstm_hidden_vb)
                elif self.lstm_layer_count == 2:
                    if off_record:
                        p_vb, sig_vb, v_vb, _, _ = self.model(state_vb, self.lstm_hidden_vb, self.lstm_hidden_vb2)
                    else:
                        p_vb, sig_vb, v_vb, self.lstm_hidden_vb, self.lstm_hidden_vb2 = self.model(state_vb,
                                                                                                   self.lstm_hidden_vb,
                                                                                                   self.lstm_hidden_vb2)
            else:
                p_vb, sig_vb, v_vb = self.model(state_vb)
            if self.training:
                _eps = torch.randn(p_vb.size())
                action = (p_vb + sig_vb.sqrt() * Variable(_eps)).data.numpy()  # TODO:?
            else:
                action = p_vb.data.numpy()
            return action, p_vb, sig_vb, v_vb
        else:
            if self.master.enable_lstm:
                if self.lstm_layer_count == 1:
                    if off_record:
                        p_vb, v_vb, _ = self.model(state_vb, self.lstm_hidden_vb)
                    else:
                        p_vb, v_vb, self.lstm_hidden_vb = self.model(state_vb, self.lstm_hidden_vb)
                elif self.lstm_layer_count == 2:
                    if off_record:
                        p_vb, v_vb, _, _ = self.model(state_vb, self.lstm_hidden_vb, self.lstm_hidden_vb2)
                    else:
                        p_vb, v_vb, self.lstm_hidden_vb, self.lstm_hidden_vb2 = self.model(state_vb,
                                                                                           self.lstm_hidden_vb,
                                                                                           self.lstm_hidden_vb2)
                extras = None
            else:
                p_vb, v_vb, extras = self.model(state_vb)
            if self.training:
                action = p_vb.multinomial().data.squeeze().numpy()
            else:
                action = p_vb.max(1)[1].data.squeeze().numpy()
            return action, p_vb, v_vb, extras

    def _normal(self, x, mu, sigma_sq):
        a = (-1 * (x - mu).pow(2) / (2 * sigma_sq)).exp()
        b = 1 / (2 * sigma_sq * self.pi_vb.expand_as(sigma_sq)).sqrt()
        return (a * b).log()


# noinspection PyPep8Naming
class A3CLearner(A3CSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning(
            "<===================================> A3C-Learner #" + str(process_id) + " {Env & Model}")
        super(A3CLearner, self).__init__(master, process_id)

        self._reset_rollout()

        self.training = True  # choose actions by polynomial
        self.model.train(self.training)
        if self.master.icm:
            self.icm_inv_model.train(self.training)
            self.icm_fwd_model.train(self.training)

        # local counters
        self.frame_step = 0  # local frame step counter
        self.train_step = 0  # local train step counter
        # local training stats
        self.p_loss_avg = 0.  # global policy loss
        self.v_loss_avg = 0.  # global value loss
        self.loss_avg = 0.  # global value loss
        self.loss_counter = 0  # storing this many losses
        self.icm_inv_loss_avg = 0.
        self.icm_fwd_loss_avg = 0.
        self.icm_inv_accuracy_avg = 0.
        self.grad_magnitude_avg = 0.
        self.grad_magnitude_max = 0.
        self._reset_training_loggings()

        # copy local training stats to global every prog_freq
        self.last_prog = time.time()

    def _reset_training_loggings(self):
        self.p_loss_avg = 0.
        self.v_loss_avg = 0.
        self.loss_avg = 0.
        self.loss_counter = 0
        self.icm_inv_loss_avg = 0.
        self.icm_fwd_loss_avg = 0.
        self.icm_inv_accuracy_avg = 0.
        self.grad_magnitude_avg = 0.
        self.grad_magnitude_max = 0.

    def _reset_rollout(self):  # for storing the experiences collected through one rollout
        self.rollout = A3C_Experience(state0=[],
                                      action=[],
                                      reward=[],
                                      state1=[],
                                      terminal1=[],
                                      policy_vb=[],
                                      sigmoid_vb=[],
                                      value0_vb=[],
                                      features0=[],
                                      features1=[])

    def _get_valueT_vb(self):
        if self.rollout.terminal1[-1]:  # for terminal sT
            valueT_vb = Variable(torch.zeros(self.master.num_robots, 1))
        else:  # for non-terminal sT
            sT_vb = self._preprocessState(self.rollout.state1[-1], True)  # bootstrap from last state
            if self.master.enable_continuous:
                if self.master.enable_lstm:
                    if self.lstm_layer_count == 1:
                        _, _, valueT_vb, _ = self.model(sT_vb, self.lstm_hidden_vb)  # NOTE: only doing inference here
                    elif self.lstm_layer_count == 2:
                        _, _, valueT_vb, _, _ = self.model(sT_vb, self.lstm_hidden_vb,
                                                           self.lstm_hidden_vb2)  # NOTE: only doing inference here
                else:
                    _, _, valueT_vb = self.model(sT_vb)  # NOTE: only doing inference here
            else:
                if self.master.enable_lstm:
                    if self.lstm_layer_count == 1:
                        _, valueT_vb, _ = self.model(sT_vb, self.lstm_hidden_vb)  # NOTE: only doing inference here
                    elif self.lstm_layer_count == 2:
                        _, valueT_vb, _, _ = self.model(sT_vb, self.lstm_hidden_vb,
                                                        self.lstm_hidden_vb2)  # NOTE: only doing inference here
                else:
                    _, valueT_vb, _ = self.model(sT_vb)  # NOTE: only doing inference here
            # NOTE: here valueT_vb.volatile=True since sT_vb.volatile=True
            # NOTE: if we use detach() here, it would remain volatile
            # NOTE: then all the follow-up computations would only give volatile loss variables
            valueT_vb = Variable(valueT_vb.data)

        return valueT_vb

    def _backward(self):
        rollout_steps = len(self.rollout.reward)

        # ICM first if enabled
        if self.master.icm:
            if self.icm_inv_model.same_features():
                _, _, _, extras = self._forward(self._preprocessState(self.experience.state1),
                                                off_record=True)
                self.rollout.features1.append(Variable(extras['features'].data))
                state_start = torch.cat(self.rollout.features0, dim=0)
                state_next = torch.cat(self.rollout.features1, dim=0)
            else:
                state_start = np.array(self.rollout.state0).reshape(-1, self.master.state_shape + 3)[:,
                              :self.master.state_shape]
                state_next = np.array(self.rollout.state1).reshape(-1, self.master.state_shape + 3)[:,
                             :self.master.state_shape]
                state_start = Variable(torch.from_numpy(state_start).type(self.master.dtype))
                state_next = Variable(torch.from_numpy(state_next).type(self.master.dtype))

            actions = np.array(self.rollout.action).reshape(-1)  # TODO: is this right for several robots?
            actions = Variable(torch.from_numpy(actions).long(), requires_grad=False)

            features, features_next, action_logits, action_probs = \
                self.icm_inv_model.forward((state_start, state_next))

            icm_inv_loss = self.icm_inv_loss_criterion(action_logits, actions)
            icm_inv_loss_mean = icm_inv_loss.mean()

            features_next_pred = self.icm_fwd_model.forward((features, actions))

            # icm_fwd_loss = 0.5 * self.icm_fwd_loss_criterion(features_next_pred, features_next).mean(dim=1)
            icm_fwd_loss = 0.5 * torch.pow(features_next_pred - features_next, 2).mean(dim=1)
            icm_fwd_loss_mean = icm_fwd_loss.mean()

            icm_loss_mean = (1 - self.master.icm_fwd_wt) * icm_inv_loss_mean \
                            + self.master.icm_fwd_wt * icm_fwd_loss_mean
            icm_loss_mean.backward()

            self.icm_inv_loss_avg += icm_inv_loss_mean.data.numpy()
            self.icm_fwd_loss_avg += icm_fwd_loss_mean.data.numpy()
            self.icm_inv_accuracy_avg += actions.eq(action_probs.max(1)[1]).sum().data.numpy()[0] / float(
                actions.size()[0])

            icm_fwd_loss_detached = Variable(icm_fwd_loss.data)

        # preparation
        policy_vb = self.rollout.policy_vb
        if self.master.enable_continuous:
            action_batch_vb = Variable(torch.from_numpy(np.array(self.rollout.action)))
            if self.master.use_cuda:
                action_batch_vb = action_batch_vb.cuda()
            sigma_vb = self.rollout.sigmoid_vb
        else:
            action_batch_vb = Variable(torch.from_numpy(np.array(self.rollout.action)).long())
            if self.master.use_cuda:
                action_batch_vb = action_batch_vb.cuda()
            policy_log_vb = [torch.log(policy_vb[i] + self.epsilon) for i in range(rollout_steps)]
            entropy_vb = [- (policy_log_vb[i] * policy_vb[i]).sum(1) for i in range(rollout_steps)]
            if hasattr(self.master, "num_robots"):
                policy_log_vb = [
                    policy_log_vb[i].gather(1, action_batch_vb[i].unsqueeze(0).view(self.master.num_robots, -1)) for i
                    in range(rollout_steps)]
            else:
                policy_log_vb = [policy_log_vb[i].gather(1, action_batch_vb[i].unsqueeze(0)) for i in
                                 range(rollout_steps)]
        valueT_vb = self._get_valueT_vb()
        self.rollout.value0_vb.append(
            Variable(valueT_vb.data))  # NOTE: only this last entry is Volatile, all others are still in the graph
        gae_ts = torch.zeros(self.master.num_robots, 1)

        # compute loss
        policy_loss_vb = 0.
        value_loss_vb = 0.
        for i in reversed(range(rollout_steps)):
            reward_vb = Variable(torch.from_numpy(self.rollout.reward[i])).float().view(-1, 1)
            if self.master.icm and self.master.icm_plus_reward:
                reward_vb += torch.clamp(self.master.icm_beta * icm_fwd_loss_detached[i], max=0.045)
            valueT_vb = self.master.gamma * valueT_vb + reward_vb
            advantage_vb = valueT_vb - self.rollout.value0_vb[i]
            value_loss_vb = value_loss_vb + 0.5 * advantage_vb.pow(2)

            # Generalized Advantage Estimation
            tderr_ts = reward_vb.data + self.master.gamma * self.rollout.value0_vb[i + 1].data - \
                       self.rollout.value0_vb[i].data
            gae_ts = self.master.gamma * gae_ts * self.master.tau + tderr_ts
            if self.master.enable_continuous:
                _log_prob = self._normal(action_batch_vb[i], policy_vb[i], sigma_vb[i])
                _entropy = 0.5 * ((sigma_vb[i] * 2 * self.pi_vb.expand_as(sigma_vb[i])).log() + 1)
                policy_loss_vb -= (_log_prob * Variable(gae_ts).expand_as(
                    _log_prob)).sum() + self.master.beta * _entropy.sum()
            else:
                policy_loss_vb -= policy_log_vb[i] * Variable(gae_ts) + (self.master.beta * entropy_vb[i]).view(
                    self.master.num_robots, -1)

        loss_vb = policy_loss_vb + 0.5 * value_loss_vb
        loss_vb = loss_vb.mean(dim=0)
        loss_vb.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.master.clip_grad)

        # targets random for each episode, each robot has its target  # DONE
        # random map for each episode  # DONE
        # update a3c code for rewards for each robot  # DONE

        self._ensure_global_grads()
        self.master.optimizer.step()
        if self.master.icm:
            self.master.icm_inv_optimizer.step()
            self.master.icm_fwd_optimizer.step()
        self.train_step += 1
        self.master.train_step.value += 1

        # adjust learning rate if enabled
        if self.master.lr_decay:
            self.master.lr_adjusted.value = max(
                self.master.lr * (self.master.steps - self.master.train_step.value) / self.master.steps, 1e-32)
            adjust_learning_rate(self.master.optimizer, self.master.lr_adjusted.value)

        # log training stats
        self.p_loss_avg += policy_loss_vb.data.numpy()
        self.v_loss_avg += value_loss_vb.data.numpy()
        self.loss_avg += loss_vb.data.numpy()
        self.grad_magnitude_avg += np.mean([np.abs(p.grad.data.norm()) for p in self.model.parameters()])
        self.grad_magnitude_max = np.max(
            [np.abs(p.grad.data.norm()) for p in self.model.parameters()] + [self.grad_magnitude_max]
        )
        self.loss_counter += 1

    def _rollout(self, episode_steps, episode_reward):
        # reset rollout experiences
        self._reset_rollout()

        t_start = self.frame_step
        # continue to rollout only if:
        # 1. not running out of max steps of this current rollout, and
        # 2. not terminal, and
        # 3. not exceeding max steps of this current episode
        # 4. master not exceeding max train steps
        while (self.frame_step - t_start) < self.master.rollout_steps \
                and not self.experience.terminal1 \
                and (self.master.early_stop is None or episode_steps < self.master.early_stop):
            # NOTE: here first store the last frame: experience.state1 as rollout.state0
            self.rollout.state0.append(self.experience.state1)
            # then get the action to take from rollout.state0 (experience.state1)
            if self.master.enable_continuous:
                action, p_vb, sig_vb, v_vb = self._forward(self._preprocessState(self.experience.state1))
                self.rollout.sigmoid_vb.append(sig_vb)
            else:
                action, p_vb, v_vb, extras = self._forward(self._preprocessState(self.experience.state1))
            # then execute action in env to get a new experience.state1 -> rollout.state1
            self.experience = self.env.step(action)
            # push experience into rollout
            self.rollout.action.append(action)
            self.rollout.reward.append(self.experience.reward)
            self.rollout.state1.append(self.experience.state1)
            self.rollout.terminal1.append(self.experience.terminal1)
            self.rollout.policy_vb.append(p_vb)
            self.rollout.value0_vb.append(v_vb)

            if self.master.icm and extras is not None and 'features' in extras:
                self.rollout.features0.append(Variable(extras['features'].data))
                if len(self.rollout.features0) > 1:
                    self.rollout.features1.append(Variable(self.rollout.features0[-1].data))

            episode_steps += 1
            episode_reward += self.experience.reward
            self.frame_step += 1
            self.master.frame_step.value += 1

            if self.master.frame_step.value % (1000 * self.master.rollout_steps) == 0:
                print("train step: {0}, frame step {1}, time: {2}".format(self.master.train_step.value,
                                                                          self.master.frame_step.value, time.time()))

            # NOTE: we put this condition in the end to make sure this current rollout won't be empty
            if self.master.train_step.value >= self.master.steps:
                break

        return episode_steps, episode_reward

    def run(self):
        # make sure processes are not completely synced by sleeping a bit
        time.sleep(int(np.random.rand() * (self.process_id + 5)))

        nepisodes = 0
        nepisodes_solved = 0
        episode_steps = None
        episode_reward = None
        should_start_new = True
        while self.master.train_step.value < self.master.steps:
            # sync in every step
            self._sync_local_with_global()
            self.model.zero_grad()
            if self.master.icm:
                self.icm_inv_model.zero_grad()
                self.icm_fwd_model.zero_grad()

            # start of a new episode
            if should_start_new:
                episode_steps = 0
                episode_reward = np.zeros(self.master.num_robots)
                # reset lstm_hidden_vb for new episode
                if self.master.enable_lstm:
                    # NOTE: clear hidden state at the beginning of each episode
                    self._reset_lstm_hidden_vb_episode()
                # Obtain the initial observation by resetting the environment
                self._reset_experience()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                # reset flag
                should_start_new = False
            if self.master.enable_lstm:
                # NOTE: detach the previous hidden variable from the graph at the beginning of each rollout
                self._reset_lstm_hidden_vb_rollout()
            # Run a rollout for rollout_steps or until terminal
            episode_steps, episode_reward = self._rollout(episode_steps, episode_reward)

            if self.experience.terminal1 or \
                            self.master.early_stop and episode_steps >= self.master.early_stop:
                nepisodes += 1
                should_start_new = True
                if self.experience.terminal1 and self.experience.reward >= 0:
                    nepisodes_solved += 1
                    self.master.terminations_count.value += 1

            # calculate loss
            self._backward()

            # copy local training stats to global at prog_freq, and clear up local stats
            if time.time() - self.last_prog >= self.master.prog_freq:
                self.master.p_loss_avg.value += self.p_loss_avg.mean()
                self.master.v_loss_avg.value += self.v_loss_avg.mean()
                self.master.loss_avg.value += self.loss_avg.mean()
                self.master.loss_counter.value += self.loss_counter
                self.master.grad_magnitude_avg.value += self.grad_magnitude_avg
                val = self.master.grad_magnitude_max.value
                self.master.grad_magnitude_max.value = max(self.grad_magnitude_max, val)
                if self.master.icm:
                    self.master.icm_inv_loss_avg.value += self.icm_inv_loss_avg
                    self.master.icm_fwd_loss_avg.value += self.icm_fwd_loss_avg
                    self.master.icm_inv_accuracy_avg.value += self.icm_inv_accuracy_avg
                self._reset_training_loggings()
                self.last_prog = time.time()


class A3CEvaluator(A3CSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> A3C-Evaluator {Env & Model}")
        super(A3CEvaluator, self).__init__(master, process_id)

        self.training = False  # choose actions w/ max probability
        self.model.train(self.training)
        if self.master.icm:
            self.icm_inv_model.train(self.training)
            self.icm_fwd_model.train(self.training)

        self._reset_loggings()

        self.start_time = time.time()
        self.last_eval = time.time()

    def _reset_loggings(self):
        # training stats across all processes
        self.p_loss_avg_log = []
        self.v_loss_avg_log = []
        self.loss_avg_log = []
        self.icm_inv_loss_avg_log = []
        self.icm_fwd_loss_avg_log = []
        self.icm_inv_accuracy_avg_log = []
        self.grad_magnitude_avg_log = []
        self.grad_magnitude_max_log = []
        # evaluation stats
        self.entropy_avg_log = []
        self.v_avg_log = []
        self.steps_avg_log = []
        self.steps_std_log = []
        self.reward_avg_log = []
        self.reward_std_log = []
        self.icm_reward_avg_log = []
        self.icm_reward_std_log = []
        self.nepisodes_log = []
        self.nepisodes_solved_log = []
        self.repisodes_solved_log = []
        self.terminals_reached_log = []
        self.action_counts = np.zeros(self.master.action_dim)
        # placeholders for windows for online curve plotting
        if self.master.visualize:
            # training stats across all processes
            self.win_p_loss_avg = "win_p_loss_avg"
            self.win_v_loss_avg = "win_v_loss_avg"
            self.win_loss_avg = "win_loss_avg"
            self.win_icm_inv_loss_avg = "win_icm_inv_loss_avg"
            self.win_icm_fwd_loss_avg = "win_icm_fwd_loss_avg"
            self.win_icm_inv_accuracy_avg = "win_icm_inv_accuracy_avg"
            self.win_grad_magnitude_avg = "win_grad_magnitude_avg"
            self.win_grad_magnitude_max = "win_grad_magnitude_max"
            # evaluation stats
            self.win_entropy_avg = "win_entropy_avg"
            self.win_v_avg = "win_v_avg"
            self.win_steps_avg = "win_steps_avg"
            self.win_steps_std = "win_steps_std"
            self.win_reward_avg = "win_reward_avg"
            self.win_reward_std = "win_reward_std"
            self.win_icm_reward_avg = "win_icm_reward_avg"
            self.win_icm_reward_std = "win_icm_reward_std"
            self.win_nepisodes = "win_nepisodes"
            self.win_nepisodes_solved = "win_nepisodes_solved"
            self.win_repisodes_solved = "win_repisodes_solved"
            self.win_terminals_reached = "win_terminals_reached"
            self.win_action_counts = "action_counts"

    def _eval_model(self):
        self.last_eval = time.time()
        eval_at_train_step = self.master.train_step.value
        eval_at_frame_step = self.master.frame_step.value
        # first grab the latest global model to do the evaluation
        self._sync_local_with_global()

        # evaluate
        eval_step = 0

        eval_entropy_log = []
        eval_v_log = []
        eval_nepisodes = 0
        eval_nepisodes_solved = 0
        eval_episode_steps = None
        eval_episode_steps_log = []
        eval_episode_reward = None
        eval_episode_reward_log = []
        eval_episode_icm_reward = None
        eval_episode_icm_reward_log = []
        eval_should_start_new = True
        eval_episode_state_history = []
        eval_episode_features_history = []
        eval_episode_action_history = []
        while eval_step < self.master.eval_steps:
            if eval_should_start_new:  # start of a new episode
                eval_episode_steps = 0
                eval_episode_reward = 0.
                eval_episode_icm_reward = 0.
                # reset lstm_hidden_vb for new episode
                if self.master.enable_lstm:
                    # NOTE: clear hidden state at the beginning of each episode
                    self._reset_lstm_hidden_vb_episode(self.training)
                # Obtain the initial observation by resetting the environment
                self._reset_experience()
                self.experience = self.env.reset()
                eval_episode_state_history.append(self.experience.state1)
                assert self.experience.state1 is not None
                if not self.training:
                    if self.master.visualize: self.env.visual()
                    if self.master.render: self.env.render()
                # reset flag
                eval_should_start_new = False
            if self.master.enable_lstm:
                # TODO: why is it here at each step?
                # NOTE: detach the previous hidden variable from the graph at the beginning of each step
                # NOTE: not necessary here in evaluation but we do it anyways
                self._reset_lstm_hidden_vb_rollout()

            # Run a single step
            if self.master.enable_continuous:
                eval_action, p_vb, sig_vb, v_vb = self._forward(self._preprocessState(self.experience.state1, True))
            else:
                eval_action, p_vb, v_vb, extras = self._forward(self._preprocessState(self.experience.state1, True))
            self.experience = self.env.step(eval_action)
            self.action_counts[eval_action] += 1

            eval_episode_state_history.append(self.experience.state1)
            if self.master.icm and extras is not None and 'features' in extras:
                eval_episode_features_history.append(Variable(extras['features'].data))
            eval_episode_action_history.append(eval_action)

            if not self.training:
                if self.master.visualize: self.env.visual()
                if self.master.render: self.env.render()
            if self.experience.terminal1 or \
                            self.master.early_stop and (eval_episode_steps + 1) == self.master.early_stop or \
                            (eval_step + 1) == self.master.eval_steps:
                eval_should_start_new = True

            eval_episode_steps += 1
            eval_episode_reward += self.experience.reward
            eval_step += 1

            if eval_should_start_new:
                eval_nepisodes += 1
                if self.experience.terminal1 and self.experience.reward >= 0:
                    eval_nepisodes_solved += 1

                if self.master.icm:
                    eval_episode_icm_reward += self.calculate_episode_icm_reward(eval_episode_action_history,
                                                                                 eval_episode_features_history,
                                                                                 eval_episode_state_history)

                # This episode is finished, report and reset
                # NOTE make no sense for continuous
                if self.master.enable_continuous:
                    eval_entropy_log.append(
                        [0.5 * ((sig_vb * 2 * self.pi_vb.expand_as(sig_vb)).log() + 1).data.numpy()])
                else:
                    eval_entropy_log.append(
                        [np.mean((-torch.log(p_vb.data.squeeze() + self.epsilon) * p_vb.data.squeeze()).numpy())])
                eval_v_log.append([v_vb.data.numpy()])
                eval_episode_steps_log.append([eval_episode_steps])
                eval_episode_reward_log.append([eval_episode_reward])
                eval_episode_icm_reward_log.append([eval_episode_icm_reward])
                self._reset_experience()
                eval_episode_state_history = []
                eval_episode_features_history = []
                eval_episode_action_history = []
                eval_episode_steps = None
                eval_episode_reward = None

        # Logging for this evaluation phase
        loss_counter = self.master.loss_counter.value
        p_loss_avg = self.master.p_loss_avg.value / loss_counter if loss_counter > 0 else 0.
        v_loss_avg = self.master.v_loss_avg.value / loss_counter if loss_counter > 0 else 0.
        loss_avg = self.master.loss_avg.value / loss_counter if loss_counter > 0 else 0.
        if self.master.icm:
            icm_inv_loss_avg = self.master.icm_inv_loss_avg.value / loss_counter if loss_counter > 0 else 0.
            icm_fwd_loss_avg = self.master.icm_fwd_loss_avg.value / loss_counter if loss_counter > 0 else 0.
            icm_inv_accuracy_avg = self.master.icm_inv_accuracy_avg.value / loss_counter if loss_counter > 0 else 0.
        grad_magnitude_avg = self.master.grad_magnitude_avg.value / loss_counter if loss_counter > 0 else 0.
        grad_magnitude_max = self.master.grad_magnitude_max.value
        self.master._reset_training_logs()

        def _log_at_step(eval_at_step):
            self.p_loss_avg_log.append([eval_at_step, p_loss_avg])
            self.v_loss_avg_log.append([eval_at_step, v_loss_avg])
            self.loss_avg_log.append([eval_at_step, loss_avg])
            if self.master.icm:
                self.icm_inv_loss_avg_log.append([eval_at_step, icm_inv_loss_avg])
                self.icm_fwd_loss_avg_log.append([eval_at_step, icm_fwd_loss_avg])
                self.icm_inv_accuracy_avg_log.append([eval_at_step, icm_inv_accuracy_avg])
            self.grad_magnitude_avg_log.append([eval_at_step, grad_magnitude_avg])
            self.grad_magnitude_max_log.append([eval_at_step, grad_magnitude_max])
            self.entropy_avg_log.append([eval_at_step, np.mean(np.asarray(eval_entropy_log))])
            self.v_avg_log.append([eval_at_step, np.mean(np.asarray(eval_v_log))])
            self.steps_avg_log.append([eval_at_step, np.mean(np.asarray(eval_episode_steps_log))])
            self.steps_std_log.append([eval_at_step, np.std(np.asarray(eval_episode_steps_log))])
            self.reward_avg_log.append([eval_at_step, np.mean(np.asarray(eval_episode_reward_log))])
            self.reward_std_log.append([eval_at_step, np.std(np.asarray(eval_episode_reward_log))])
            if self.master.icm:
                self.icm_reward_avg_log.append([eval_at_step, np.mean(np.asarray(eval_episode_icm_reward_log))])
                self.icm_reward_std_log.append([eval_at_step, np.std(np.asarray(eval_episode_icm_reward_log))])
            self.nepisodes_log.append([eval_at_step, eval_nepisodes])
            self.nepisodes_solved_log.append([eval_at_step, eval_nepisodes_solved])
            self.repisodes_solved_log.append(
                [eval_at_step, (eval_nepisodes_solved / eval_nepisodes) if eval_nepisodes > 0 else 0.])
            self.terminals_reached_log.append([self.master.train_step.value, self.master.terminations_count.value])
            # logging
            self.master.logger.warning("Reporting       @ Step: " + str(eval_at_step) + " | Elapsed Time: " + str(
                time.time() - self.start_time))
            self.master.logger.warning("Iteration: {}; lr: {}".format(eval_at_step, self.master.lr_adjusted.value))
            self.master.logger.warning("Iteration: {}; p_loss_avg: {}".format(eval_at_step, self.p_loss_avg_log[-1][1]))
            self.master.logger.warning("Iteration: {}; v_loss_avg: {}".format(eval_at_step, self.v_loss_avg_log[-1][1]))
            self.master.logger.warning("Iteration: {}; loss_avg: {}".format(eval_at_step, self.loss_avg_log[-1][1]))
            if self.master.icm:
                self.master.logger.warning(
                    "Iteration: {}; icm_inv_loss_avg: {}".format(eval_at_step, self.icm_inv_loss_avg_log[-1][1]))
                self.master.logger.warning(
                    "Iteration: {}; icm_fwd_loss_avg: {}".format(eval_at_step, self.icm_fwd_loss_avg_log[-1][1]))
                self.master.logger.warning(
                    "Iteration: {}; icm_inv_accuracy_avg: {}".format(eval_at_step,
                                                                     self.icm_inv_accuracy_avg_log[-1][1]))
            self.master.logger.warning(
                "Iteration: {}; grad_magnitude_avg: {}".format(eval_at_step, self.grad_magnitude_avg_log[-1][1]))
            self.master.logger.warning(
                "Iteration: {}; grad_magnitude_max: {}".format(eval_at_step, self.grad_magnitude_max_log[-1][1]))
            self.master._reset_training_logs()
            self.master.logger.warning(
                "Evaluating      @ Step: " + str(eval_at_train_step) + " | (" + str(eval_at_frame_step) + " frames)...")
            self.master.logger.warning("Evaluation        Took: " + str(time.time() - self.last_eval))
            self.master.logger.warning(
                "Iteration: {}; entropy_avg: {}".format(eval_at_step, self.entropy_avg_log[-1][1]))
            self.master.logger.warning("Iteration: {}; v_avg: {}".format(eval_at_step, self.v_avg_log[-1][1]))
            self.master.logger.warning("Iteration: {}; steps_avg: {}".format(eval_at_step, self.steps_avg_log[-1][1]))
            self.master.logger.warning("Iteration: {}; steps_std: {}".format(eval_at_step, self.steps_std_log[-1][1]))
            self.master.logger.warning("Iteration: {}; reward_avg: {}".format(eval_at_step, self.reward_avg_log[-1][1]))
            self.master.logger.warning("Iteration: {}; reward_std: {}".format(eval_at_step, self.reward_std_log[-1][1]))
            if self.master.icm:
                self.master.logger.warning(
                    "Iteration: {}; icm_reward_avg: {}".format(eval_at_step, self.icm_reward_avg_log[-1][1]))
                self.master.logger.warning(
                    "Iteration: {}; icm_reward_std: {}".format(eval_at_step, self.icm_reward_std_log[-1][1]))
            self.master.logger.warning("Iteration: {}; nepisodes: {}".format(eval_at_step, self.nepisodes_log[-1][1]))
            self.master.logger.warning(
                "Iteration: {}; nepisodes_solved: {}".format(eval_at_step, self.nepisodes_solved_log[-1][1]))
            self.master.logger.warning(
                "Iteration: {}; repisodes_solved: {}".format(eval_at_step, self.repisodes_solved_log[-1][1]))
            self.master.logger.warning(
                "Iteration: {}; terminals_reached: {}".format(eval_at_step, self.terminals_reached_log[-1][1]))
            self.master.logger.warning(
                "Iteration: {}; action_counts: {}".format(eval_at_step, self.action_counts))

        if self.master.enable_log_at_train_step:
            _log_at_step(eval_at_train_step)
        else:
            _log_at_step(eval_at_frame_step)

        # plotting
        if self.master.visualize:
            self.win_p_loss_avg = self.master.vis.scatter(X=np.array(self.p_loss_avg_log), env=self.master.refs,
                                                          win=self.win_p_loss_avg, opts=dict(title="p_loss_avg"))
            self.win_v_loss_avg = self.master.vis.scatter(X=np.array(self.v_loss_avg_log), env=self.master.refs,
                                                          win=self.win_v_loss_avg, opts=dict(title="v_loss_avg"))
            self.win_loss_avg = self.master.vis.scatter(X=np.array(self.loss_avg_log), env=self.master.refs,
                                                        win=self.win_loss_avg, opts=dict(title="loss_avg"))

            if self.master.icm:
                self.win_icm_inv_loss_avg = self.master.vis.scatter(X=np.array(self.icm_inv_loss_avg_log),
                                                                    env=self.master.refs,
                                                                    win=self.win_icm_inv_loss_avg,
                                                                    opts=dict(title="icm_inv_loss_avg"))
                self.win_icm_fwd_loss_avg = self.master.vis.scatter(X=np.array(self.icm_fwd_loss_avg_log),
                                                                    env=self.master.refs,
                                                                    win=self.win_icm_fwd_loss_avg,
                                                                    opts=dict(title="icm_fwd_loss_avg"))
                self.win_icm_inv_accuracy_avg = self.master.vis.scatter(X=np.array(self.icm_inv_accuracy_avg_log),
                                                                        env=self.master.refs,
                                                                        win=self.win_icm_inv_accuracy_avg,
                                                                        opts=dict(title="icm_inv_accuracy_avg"))
            self.win_grad_magnitude_avg = self.master.vis.scatter(X=np.array(self.grad_magnitude_avg_log),
                                                                  env=self.master.refs,
                                                                  win=self.win_grad_magnitude_avg,
                                                                  opts=dict(title="grad_magnitude_avg"))
            self.win_grad_magnitude_max = self.master.vis.scatter(X=np.array(self.grad_magnitude_max_log),
                                                                  env=self.master.refs,
                                                                  win=self.win_grad_magnitude_max,
                                                                  opts=dict(title="grad_magnitude_max_avg"))

            self.win_entropy_avg = self.master.vis.scatter(X=np.array(self.entropy_avg_log), env=self.master.refs,
                                                           win=self.win_entropy_avg, opts=dict(title="entropy_avg"))
            self.win_v_avg = self.master.vis.scatter(X=np.array(self.v_avg_log), env=self.master.refs,
                                                     win=self.win_v_avg, opts=dict(title="v_avg"))
            self.win_steps_avg = self.master.vis.scatter(X=np.array(self.steps_avg_log), env=self.master.refs,
                                                         win=self.win_steps_avg, opts=dict(title="steps_avg"))
            # self.win_steps_std = self.master.vis.scatter(X=np.array(self.steps_std_log), env=self.master.refs,
            #                                              win=self.win_steps_std, opts=dict(title="steps_std"))
            self.win_reward_avg = self.master.vis.scatter(X=np.array(self.reward_avg_log), env=self.master.refs,
                                                          win=self.win_reward_avg, opts=dict(title="reward_avg"))
            # self.win_reward_std = self.master.vis.scatter(X=np.array(self.reward_std_log), env=self.master.refs,
            #                                               win=self.win_reward_std, opts=dict(title="reward_std"))
            if self.master.icm:
                self.win_icm_reward_avg = self.master.vis.scatter(X=np.array(self.icm_reward_avg_log),
                                                                  env=self.master.refs,
                                                                  win=self.win_icm_reward_avg,
                                                                  opts=dict(title="icm_reward_avg"))
                # self.win_icm_reward_std = self.master.vis.scatter(X=np.array(self.icm_reward_std_log),
                #                                                   env=self.master.refs, win=self.win_icm_reward_std,
                #                                                   opts=dict(title="icm_reward_std"))

            self.win_nepisodes = self.master.vis.scatter(X=np.array(self.nepisodes_log), env=self.master.refs,
                                                         win=self.win_nepisodes, opts=dict(title="nepisodes"))
            self.win_nepisodes_solved = self.master.vis.scatter(X=np.array(self.nepisodes_solved_log),
                                                                env=self.master.refs, win=self.win_nepisodes_solved,
                                                                opts=dict(title="nepisodes_solved"))
            self.win_repisodes_solved = self.master.vis.scatter(X=np.array(self.repisodes_solved_log),
                                                                env=self.master.refs, win=self.win_repisodes_solved,
                                                                opts=dict(title="repisodes_solved"))
            self.win_terminals_reached = self.master.vis.scatter(X=np.array(self.terminals_reached_log),
                                                                 env=self.master.refs, win=self.win_terminals_reached,
                                                                 opts=dict(title="terminals_reached"))
            self.win_action_counts = self.master.vis.bar(X=self.action_counts, env=self.master.refs,
                                                         win=self.win_action_counts,
                                                         opts=dict(title="action_counts"))

        self.last_eval = time.time()

        # save model
        self.master._save_model(eval_at_train_step, self.reward_avg_log[-1][1])
        if self.master.icm:
            self.master._save_icm_models(eval_at_train_step,
                                         self.icm_inv_loss_avg_log[-1][1], self.icm_fwd_loss_avg_log[-1][1])

    def calculate_episode_icm_reward(self, eval_episode_action_history, eval_episode_features_history,
                                     eval_episode_state_history):
        if self.icm_inv_model.same_features():
            _, _, _, extras = self._forward(self._preprocessState(self.experience.state1),
                                            off_record=True)
            eval_episode_features_history.append(Variable(extras['features'].data))
            state_start = torch.cat(eval_episode_features_history[0:-1], dim=0)
            state_next = torch.cat(eval_episode_features_history[1:], dim=0)
        else:
            state_start = np.array(eval_episode_state_history[0:-1]).reshape(-1,
                                                                             self.master.state_shape + 3)[:,
                          :self.master.state_shape]
            state_next = np.array(eval_episode_state_history[1:]).reshape(-1, self.master.state_shape + 3)[
                         :,
                         :self.master.state_shape]
            state_start = Variable(torch.from_numpy(state_start).type(self.master.dtype))
            state_next = Variable(torch.from_numpy(state_next).type(self.master.dtype))
        # TODO: is this right for several robots?
        actions = np.array(eval_episode_action_history).reshape(-1)
        actions = Variable(torch.from_numpy(actions).long(), requires_grad=False)

        features, features_next, action_logits, action_probs = \
            self.icm_inv_model.forward((state_start, state_next))

        features_next_pred = self.icm_fwd_model.forward((features, actions))
        icm_fwd_loss = 0.5 * torch.pow(features_next_pred - features_next, 2).mean(dim=1)

        # TODO: ICM code has too many repetitions, bad design, error prone
        return torch.clamp(self.master.icm_beta * icm_fwd_loss, max=0.045).sum().data.numpy()[0]

    def run(self):
        while self.master.train_step.value < self.master.steps:
            if time.time() - self.last_eval > self.master.eval_freq:
                self._eval_model()
        # we also do a final evaluation after training is done
        self._eval_model()


class A3CTester(A3CSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> A3C-Tester {Env & Model}")
        super(A3CTester, self).__init__(master, process_id)

        self.episode_experience_history = []
        self.episode_features_history = []
        self.action_history = []
        self.episode_icm_reward = 0.
        if self.master.plot_icm_test:
            assert self.master.icm, 'Asked to plot icm features when ICM is off'
            self.map_image = self.env.read_static_map_image()
            plt.rc('font', family='Times New Roman')
            matplotlib.rcParams.update({'font.size': 16})
            self.triple_icm_test = True
            if self.triple_icm_test:
                self.icm_inv_model2 = self.master.icm_inv_model_prototype(self.master.model_params)
                self.icm_fwd_model2 = self.master.icm_fwd_model_prototype(self.master.model_params)
                self.icm_inv_model3 = self.master.icm_inv_model_prototype(self.master.model_params)
                self.icm_fwd_model3 = self.master.icm_fwd_model_prototype(self.master.model_params)
                inv_base = self.master.icm_inv_model_file[:self.master.icm_inv_model_file.index('step')]
                fwd_base = self.master.icm_fwd_model_file[:self.master.icm_fwd_model_file.index('step')]
                self.icm_inv_model2.load_state_dict(torch.load(inv_base + 'step_' + '115896' + '.pth'))
                self.icm_fwd_model2.load_state_dict(torch.load(fwd_base + 'step_' + '115896' + '.pth'))
                # self.icm_inv_model3.load_state_dict(torch.load(inv_base + 'step_' + '60109' + '.pth'))
                # self.icm_fwd_model3.load_state_dict(torch.load(fwd_base + 'step_' + '60109' + '.pth'))
                self.icm_inv_model3.load_state_dict(torch.load(inv_base + 'step_' + '2683' + '.pth'))
                self.icm_fwd_model3.load_state_dict(torch.load(fwd_base + 'step_' + '2683' + '.pth'))

        if self.master.plot_env_picture:
            self.map_image = self.env.read_static_map_image()
            plt.rc('font', family='Times New Roman')

        self.training = False  # choose actions w/ max probability
        # self.training = True  # choose actions by polynomial (?)
        self.model.train(self.training)
        if self.master.icm:
            self.icm_inv_model.train(self.training)
            self.icm_fwd_model.train(self.training)

        self._reset_loggings()

        self.start_time = time.time()

    # TODO: add terminations count to the log here too?
    # TODO: add ICM logs here?
    def _reset_loggings(self):
        # testing stats
        self.steps_avg_log = []
        self.steps_std_log = []
        self.reward_avg_log = []
        self.reward_std_log = []
        self.nepisodes_log = []
        self.nepisodes_solved_log = []
        self.repisodes_solved_log = []
        # placeholders for windows for online curve plotting
        if self.master.visualize:
            # evaluation stats
            self.win_steps_avg = "win_steps_avg"
            self.win_steps_std = "win_steps_std"
            self.win_reward_avg = "win_reward_avg"
            self.win_reward_std = "win_reward_std"
            self.win_nepisodes = "win_nepisodes"
            self.win_nepisodes_solved = "win_nepisodes_solved"
            self.win_repisodes_solved = "win_repisodes_solved"

    def run(self):
        test_step = 0
        test_nepisodes = 0
        test_nepisodes_solved = 0
        test_episode_steps = None
        test_episode_steps_log = []
        test_episode_reward = None
        test_episode_reward_log = []
        test_should_start_new = True
        while test_nepisodes < self.master.test_nepisodes:
            if test_should_start_new:  # start of a new episode
                test_episode_steps = 0
                test_episode_reward = 0.
                # reset lstm_hidden_vb for new episode
                if self.master.enable_lstm:
                    # NOTE: clear hidden state at the beginning of each episode
                    self._reset_lstm_hidden_vb_episode(self.training)
                # Obtain the initial observation by resetting the environment
                self._reset_experience()
                self.experience = self.env.reset()
                while self.experience.terminal1:
                    self.experience = self.env.reset()
                self.episode_experience_history.append(self.experience)
                assert self.experience.state1 is not None
                if not self.training:
                    if self.master.visualize: self.env.visual()
                    if self.master.render: self.env.render()
                # reset flag
                test_should_start_new = False
            if self.master.enable_lstm:
                # TODO: why is it here at each step?
                # NOTE: detach the previous hidden variable from the graph at the beginning of each step
                # NOTE: not necessary here in testing but we do it anyways
                self._reset_lstm_hidden_vb_rollout()

            # Run a single step
            if self.master.enable_continuous:
                test_action, p_vb, sig_vb, v_vb = self._forward(self._preprocessState(self.experience.state1, True))
            else:
                test_action, p_vb, v_vb, extras = self._forward(self._preprocessState(self.experience.state1, True))
            self.experience = self.env.step(test_action)

            self.episode_experience_history.append(self.experience)
            if self.master.icm and extras is not None and 'features' in extras:
                self.episode_features_history.append(Variable(extras['features'].data))
            self.action_history.append(test_action)

            if self.master.icm:
                features_next, features_next_pred = self.calculate_icm()
                icm_fwd_loss = 0.5 * torch.pow(features_next_pred - features_next, 2).mean(dim=1)

                self.episode_icm_reward += \
                    torch.clamp(self.master.icm_beta * icm_fwd_loss, max=0.045).sum().data.numpy()[0]
                if self.master.verbose_test:
                    print('total ICM reward: ', self.episode_icm_reward)

            if not self.training:
                if self.master.visualize: self.env.visual()
                if self.master.render: self.env.render()
                if self.master.plot_icm_test and test_nepisodes == 7:
                    if self.triple_icm_test:
                        self.plot_icm_test_triple(p_vb, test_nepisodes)
                    else:
                        self.plot_icm_test(p_vb, test_nepisodes)
                if self.master.plot_env_picture and test_nepisodes == 12:
                    self.plot_env_picture(test_nepisodes)

            if self.experience.terminal1 or \
                            self.master.early_stop and (test_episode_steps + 1) == self.master.early_stop:
                test_should_start_new = True

            test_episode_steps += 1
            test_episode_reward += self.experience.reward
            test_step += 1

            if test_should_start_new:
                test_nepisodes += 1
                if self.experience.terminal1 and self.experience.reward >= 0:
                    test_nepisodes_solved += 1

                # This episode is finished, report and reset
                test_episode_steps_log.append([test_episode_steps])
                test_episode_reward_log.append([test_episode_reward])
                self._reset_experience()
                self.episode_experience_history = []
                self.episode_features_history = []
                self.action_history = []
                self.episode_icm_reward = 0.
                test_episode_steps = None
                test_episode_reward = None

        self.steps_avg_log.append([test_nepisodes, np.mean(np.asarray(test_episode_steps_log))])
        self.steps_std_log.append([test_nepisodes, np.std(np.asarray(test_episode_steps_log))])
        del test_episode_steps_log
        self.reward_avg_log.append([test_nepisodes, np.mean(np.asarray(test_episode_reward_log))])
        self.reward_std_log.append([test_nepisodes, np.std(np.asarray(test_episode_reward_log))])
        del test_episode_reward_log
        self.nepisodes_log.append([test_nepisodes, test_nepisodes])
        self.nepisodes_solved_log.append([test_nepisodes, test_nepisodes_solved])
        self.repisodes_solved_log.append(
            [test_nepisodes, (test_nepisodes_solved / test_nepisodes) if test_nepisodes > 0 else 0.])
        # plotting
        if self.master.visualize:
            self.win_steps_avg = self.master.vis.scatter(X=np.array(self.steps_avg_log), env=self.master.refs,
                                                         win=self.win_steps_avg, opts=dict(title="steps_avg"))
            # self.win_steps_std = self.master.vis.scatter(X=np.array(self.steps_std_log), env=self.master.refs, win=self.win_steps_std, opts=dict(title="steps_std"))
            self.win_reward_avg = self.master.vis.scatter(X=np.array(self.reward_avg_log), env=self.master.refs,
                                                          win=self.win_reward_avg, opts=dict(title="reward_avg"))
            # self.win_reward_std = self.master.vis.scatter(X=np.array(self.reward_std_log), env=self.master.refs, win=self.win_reward_std, opts=dict(title="reward_std"))
            self.win_nepisodes = self.master.vis.scatter(X=np.array(self.nepisodes_log), env=self.master.refs,
                                                         win=self.win_nepisodes, opts=dict(title="nepisodes"))
            self.win_nepisodes_solved = self.master.vis.scatter(X=np.array(self.nepisodes_solved_log),
                                                                env=self.master.refs, win=self.win_nepisodes_solved,
                                                                opts=dict(title="nepisodes_solved"))
            self.win_repisodes_solved = self.master.vis.scatter(X=np.array(self.repisodes_solved_log),
                                                                env=self.master.refs, win=self.win_repisodes_solved,
                                                                opts=dict(title="repisodes_solved"))
        # logging
        self.master.logger.warning("Testing  Took: " + str(time.time() - self.start_time))
        self.master.logger.warning("Testing: steps_avg: {}".format(self.steps_avg_log[-1][1]))
        self.master.logger.warning("Testing: steps_std: {}".format(self.steps_std_log[-1][1]))
        self.master.logger.warning("Testing: reward_avg: {}".format(self.reward_avg_log[-1][1]))
        self.master.logger.warning("Testing: reward_std: {}".format(self.reward_std_log[-1][1]))
        self.master.logger.warning("Testing: nepisodes: {}".format(self.nepisodes_log[-1][1]))
        self.master.logger.warning("Testing: nepisodes_solved: {}".format(self.nepisodes_solved_log[-1][1]))
        self.master.logger.warning("Testing: repisodes_solved: {}".format(self.repisodes_solved_log[-1][1]))

    def plot_icm_test_triple(self, p_vb, test_nepisodes):
        features_next, features_next_pred = self.calculate_icm()
        features_next2, features_next_pred2 = self.calculate_icm(inv_model=self.icm_inv_model2,
                                                                 fwd_model=self.icm_fwd_model2)
        features_next3, features_next_pred3 = self.calculate_icm(inv_model=self.icm_inv_model3,
                                                                 fwd_model=self.icm_fwd_model3)

        fig = plt.figure(figsize=(9, 10))
        ax1 = plt.subplot2grid((10, 6), (0, 0), colspan=4, rowspan=4)
        ax2 = plt.subplot2grid((10, 6), (0, 4), colspan=2, rowspan=4)
        ax3 = plt.subplot2grid((10, 6), (4, 0), colspan=6, rowspan=2)
        ax4 = plt.subplot2grid((10, 6), (6, 0), colspan=6, rowspan=2)
        ax5 = plt.subplot2grid((10, 6), (8, 0), colspan=6, rowspan=2)

        ax1.set_anchor('W')
        self.plot_current_step(ax1, draw_path=True, draw_rays=True, show_legend=False)
        # ax1.set_title('Top view')

        actions = ('Fwd', 'Left', 'Right')  # TODO: check left and right
        y_pos = np.arange(len(actions))
        prob = p_vb.data.numpy().reshape(-1)

        ax2.barh(y_pos, prob, align='center', height=0.4)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(actions)
        ax2.set_xlim([-0.05, 1.05])
        ax2.yaxis.tick_right()
        ax2.xaxis.tick_top()
        # ax2.tick_params(axis='x', pad=-25)
        ax2.tick_params(axis='x', labelbottom='off', labeltop='off')
        ax2.tick_params(direction='in')
        # ax2.set_title('Action', y=1.10)

        x_pos = np.arange(16)
        val_f_hat = features_next_pred.data.numpy().reshape(-1)
        val_f = features_next.data.numpy().reshape(-1)

        val_f_hat = np.tanh(val_f_hat)
        val_f = np.tanh(val_f)

        ax3.bar(x_pos, val_f_hat, width=0.2, label=r'$\^\phi_{3mil}$')
        ax3.bar(x_pos + 0.2, val_f, width=0.2, label=r'$\phi_{3mil}$')
        ax3.set_ylim([-1, 1])
        ax3.set_xticks(np.arange(16))
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.tick_params(direction='in')
        ax3.tick_params(labelbottom='off', labeltop='off', labelleft='off')

        val_f_hat = features_next_pred2.data.numpy().reshape(-1)
        val_f = features_next2.data.numpy().reshape(-1)

        val_f_hat = np.tanh(val_f_hat)
        val_f = np.tanh(val_f)

        ax4.bar(x_pos, val_f_hat, width=0.2, label=r'$\^\phi_{100k}$')
        ax4.bar(x_pos + 0.2, val_f, width=0.2, label=r'$\phi_{100k}$')
        ax4.set_ylim([-1, 1])
        ax4.set_xticks(np.arange(16))
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax4.tick_params(direction='in')
        ax4.tick_params(labelbottom='off', labeltop='off', labelleft='off')

        val_f_hat = features_next_pred3.data.numpy().reshape(-1)
        val_f = features_next3.data.numpy().reshape(-1)

        val_f_hat = np.tanh(val_f_hat)
        val_f = np.tanh(val_f)

        ax5.bar(x_pos, val_f_hat, width=0.2, label=r'$\^\phi_{3k}$')
        ax5.bar(x_pos + 0.2, val_f, width=0.2, label=r'$\phi_{3k}$')
        ax5.set_ylim([-1, 1])
        ax5.set_xticks(np.arange(16))
        ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax5.tick_params(direction='in')
        ax5.tick_params(labelbottom='off', labeltop='off', labelleft='off')

        plot_dir = os.path.join('imgs', 'icm_test_triple_plots', os.path.basename(self.master.icm_fwd_model_file)[:-4])
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(
            os.path.join(plot_dir, 'episode{0:03}_step{1:03}'.format(
                test_nepisodes + 1,
                len(self.action_history)
            )),
            dpi=200,
            bbox_inches='tight',
            pad_inches=0.05
        )
        plt.close(fig)

    def plot_icm_test(self, p_vb, test_nepisodes):
        features_next, features_next_pred = self.calculate_icm()

        fig = plt.figure(figsize=(9, 6))
        ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=4, rowspan=4)
        ax2 = plt.subplot2grid((6, 6), (0, 4), colspan=2, rowspan=4)
        ax3 = plt.subplot2grid((6, 6), (4, 0), colspan=6, rowspan=2)

        ax1.set_anchor('W')
        self.plot_current_step(ax1, draw_path=True, draw_rays=True, show_legend=False)
        # ax1.set_title('Top view')

        actions = ('Fwd', 'Left', 'Right')  # TODO: check left and right
        y_pos = np.arange(len(actions))
        prob = p_vb.data.numpy().reshape(-1)

        ax2.barh(y_pos, prob, align='center', height=0.4)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(actions)
        ax2.set_xlim([-0.05, 1.05])
        ax2.yaxis.tick_right()
        ax2.xaxis.tick_top()
        # ax2.tick_params(axis='x', pad=-25)
        ax2.tick_params(axis='x', labelbottom='off', labeltop='off')
        ax2.tick_params(direction='in')
        # ax2.set_title('Action', y=1.10)

        x_pos = np.arange(16)
        val_f_hat = features_next_pred.data.numpy().reshape(-1)
        val_f = features_next.data.numpy().reshape(-1)

        val_f_hat = np.tanh(val_f_hat)
        val_f = np.tanh(val_f)

        ax3.bar(x_pos, val_f_hat, width=0.2, label=r'$\^\phi$')
        ax3.bar(x_pos + 0.2, val_f, width=0.2, label=r'$\phi$')
        ax3.set_ylim([-1, 1])
        ax3.set_xticks(np.arange(16))
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.tick_params(direction='in')
        ax3.tick_params(labelbottom='off', labeltop='off', labelleft='off')
        # ax3.set_title('ICM features')

        # plt.tight_layout()
        left = 0  # 0.125  # the left side of the subplots of the figure
        right = 1  # 0.9  # the right side of the subplots of the figure
        bottom = 0  # 0.1  # the bottom of the subplots of the figure
        top = 1  # 0.9  # the top of the subplots of the figure
        wspace = 0.1  # 0.2  # the amount of width reserved for blank space between subplots
        hspace = 0.1  # 0.2  # the amount of height reserved for white space between subplots
        # plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
        plot_dir = os.path.join('imgs', 'icm_test_plots', os.path.basename(self.master.icm_fwd_model_file)[:-4])
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        # extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(
            os.path.join(plot_dir, 'episode{0:03}_step{1:03}'.format(
                test_nepisodes + 1,
                len(self.action_history)
            )),
            dpi=200,
            # bbox_inches=extent.expanded(1.43, 1),
            bbox_inches='tight',
            pad_inches=0.05
        )
        plt.close(fig)

    def plot_env_picture(self, test_nepisodes):
        fig = plt.figure(figsize=(self.map_image.shape[1] / 100.0, self.map_image.shape[0] / 100.0))
        ax = fig.add_subplot(111)

        plt.axis('off')
        self.plot_current_step(ax, draw_path=True, draw_rays=True, show_legend=False)

        plt.tight_layout()
        plot_dir = os.path.join('imgs', 'env_pictures', os.path.basename(self.master.model_file)[:-4])
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.axis('off')
        plt.gcf().set_size_inches(self.map_image.shape[1] / 100.0, self.map_image.shape[0] / 100.0)
        plt.savefig(
            os.path.join(plot_dir, 'episode{0:03}_step{1:03}.png'.format(
                test_nepisodes + 1,
                len(self.action_history)
            )),
            dpi=300, bbox_inches='tight', pad_inches=0
        )
        plt.close(fig)

    def plot_current_step(self, ax, draw_path=False, draw_rays=False, show_legend=False):
        y_max = self.map_image.shape[0]  # the y coordinate needs to be flipped because of display (TODO) differences
        target_extras = self.episode_experience_history[0].extras
        extras = self.episode_experience_history[-1].extras
        ax.imshow(self.map_image, cmap='gray')

        target_circle = Circle(
            (target_extras['target_map_x'][0], y_max - target_extras['target_map_y'][0]),
            radius=target_extras['target_radius'] * 100, label='target'
        )
        ax.add_patch(target_circle)

        robot_pos = [extras['robot_map_x'][0], y_max - extras['robot_map_y'][0]]
        rad = extras['robot_theta'][0]
        angle = -rad * 180 / np.pi
        robot_rect = Rectangle(
            (robot_pos[0] - 10, robot_pos[1] - 10),
            20,  # width
            20,  # height
            fill=False,
            color='green',
            label='robot'
        )
        t_start = ax.transData
        t = Affine2D().rotate_deg_around(robot_pos[0], robot_pos[1], angle)
        t_end = t + t_start
        robot_rect.set_transform(t_end)
        ax.add_patch(
            robot_rect
        )

        end = [robot_pos[0] + 13 * math.cos(rad), robot_pos[1] - 13 * math.sin(rad)]
        ax.plot([robot_pos[0], end[0]], [robot_pos[1], end[1]], color='red')
        ax.set_axis_off()

        if draw_path:
            path = ax.plot(
                [state.extras['robot_map_x'][0] for state in self.episode_experience_history],
                [y_max - state.extras['robot_map_y'][0] for state in self.episode_experience_history],
                color='xkcd:water blue', linewidth=1, label='path', alpha=0.5, zorder=1
            )
            start = ax.scatter(
                [self.episode_experience_history[0].extras['robot_map_x'][0]],
                [y_max - self.episode_experience_history[0].extras['robot_map_y'][0]],
                color='green', s=4, label='start', zorder=2
            )

        if draw_rays:
            def update_angle(angle_, increment):
                angle_ += increment

                if angle_ > np.pi * 2:
                    angle_ -= np.pi * 2
                return angle_

            current_angle = self.episode_experience_history[-1].extras['robot_theta'][0]
            current_angle = update_angle(current_angle, -np.pi / 2.0)
            ranges = self.episode_experience_history[-1].state1[0, :self.master.state_shape]
            for i in range(0, self.master.state_shape, 2):
                laser = ax.plot(
                    [robot_pos[0], robot_pos[0] + ranges[i] * 100 * math.cos(current_angle)],
                    [robot_pos[1], robot_pos[1] - ranges[i] * 100 * math.sin(current_angle)],
                    color='xkcd:strawberry', linewidth=0.4, label='laser' if i == 0 else None
                )
                ax.scatter(
                    [robot_pos[0] + ranges[i] * 100 * math.cos(current_angle)],
                    [robot_pos[1] - ranges[i] * 100 * math.sin(current_angle)],
                    marker='*', s=0.4, color='xkcd:strawberry'
                )
                current_angle = update_angle(current_angle, np.pi / (self.master.state_shape - 1) * 2)

        if show_legend:
            import matplotlib.lines as mlines
            path_line = mlines.Line2D([], [], color='xkcd:water blue')
            laser_line = mlines.Line2D([], [], color='xkcd:strawberry')
            ax.legend([start, target_circle, robot_rect, laser_line, path_line],
                      ['start', 'target', 'robot', 'laser', 'path'],
                      handler_map={Circle: HandlerCircle(),
                                   Rectangle: HandlerRectangle()})

    # TODO: use elsewhere, better name
    def calculate_icm(self, inv_model=None, fwd_model=None):
        if inv_model is None:
            inv_model = self.icm_inv_model
        if fwd_model is None:
            fwd_model = self.icm_fwd_model

        if self.icm_inv_model.same_features():
            state_start = self.episode_features_history[-1]
            # TODO: correct?
            _, _, _, extras = self._forward(self._preprocessState(self.episode_experience_history[-1].state1),
                                            off_record=True)
            if extras is not None and 'features' in extras:
                state_next = Variable(extras['features'].data)
            else:
                raise Exception('ICM running with A3C features, but the A3C model does not provide them as output')
        else:
            state_start = self.episode_experience_history[-2].state1.reshape(-1, self.master.state_shape + 3)[:,
                          :self.master.state_shape]
            state_next = self.episode_experience_history[-1].state1.reshape(-1, self.master.state_shape + 3)[:,
                         :self.master.state_shape]
            state_start = Variable(torch.from_numpy(state_start).type(self.master.dtype))
            state_next = Variable(torch.from_numpy(state_next).type(self.master.dtype))

        actions = np.array(self.action_history[-1]).reshape(-1)
        actions = Variable(torch.from_numpy(actions).long(), requires_grad=False)

        features, features_next, action_logits, action_probs = \
            inv_model.forward((state_start, state_next))

        features_next_pred = fwd_model.forward((features, actions))

        return features_next, features_next_pred
