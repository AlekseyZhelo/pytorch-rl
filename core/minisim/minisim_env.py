from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import roslaunch

from core.env import Env
from core.minisim.minisim_client import MinisimClient
from utils.helpers import Experience
from utils.options import EnvParams


class MinisimEnv(Env):
    roslaunch_map_server = None
    roslaunch_node_starter = None

    def __init__(self, args, env_ind=0):
        tmp = self._reset_experience
        self._reset_experience = lambda: None
        super(MinisimEnv, self).__init__(args, env_ind)
        self._reset_experience = tmp

        assert self.env_type == "minisim"

        self.num_robots = args.num_robots
        self.curriculum = args.curriculum if hasattr(args, "curriculum") else False

        print('curriculum', self.curriculum)

        self.sim_name = 'sim' + str(self.ind)

        if MinisimEnv.roslaunch_map_server is None:
            self.init_roslaunch()

        self.node = self.launch_node()
        self.client = MinisimClient(self.num_robots, self.seed, self.curriculum, '/' + self.sim_name, self.logger)
        self.client.setup()  # TODO: move to client's init?

        # action space setup  # [linear velocity, angular velocity]
        # self.actions = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]  # ,[1, 1], [1, -1], [-1, 1], [-1, -1]]
        self.actions = [[0, 0], [10, 0], [-10, 0], [0, 16], [0, -16]]  # ,[1, 1], [1, -1], [-1, 1], [-1, -1]]
        self.logger.warning("Action Space: %s", self.actions)

        # state space setup
        self.logger.warning("State  Space: %s", self.state_shape)

        # continuous space
        if args.agent_type == "a3c":
            self.enable_continuous = args.enable_continuous
            if args.enable_continuous:
                self.logger.warning("Continuous actions not implemented for minisim yet")
        else:
            self.enable_continuous = False

        if hasattr(args, "hist_len"):
            self.hist_len = args.hist_len
            self.state_buffer = np.zeros((self.hist_len, self.state_shape + 2 * self.num_robots))
        else:
            self.hist_len = 1

        self._reset_experience()

    def _preprocessState(self, state):  # NOTE: here no preprocessing is needed
        return state

    def _reset_experience(self):
        super(MinisimEnv, self)._reset_experience()
        if self.hist_len > 1:
            self.state_buffer[:] = 0

    def _append_to_history(self, state):
        for i in range(self.state_buffer.shape[0] - 1):
            self.state_buffer[i, :] = self.state_buffer[i + 1, :]
        self.state_buffer[-1, :] = state

    @property
    def state_shape(self):
        return self.client.state_shape

    @property
    def action_dim(self):
        return len(self.actions) ** self.num_robots

    def render(self):
        self.logger.warning("WARNING: asked to render minisim - user rviz instead")

    def visual(self):
        pass

    def sample_random_action(self):  # TODO: unused
        return [self.actions[np.random.randint(0, len(self.actions))] for _ in xrange(self.num_robots)]

    def _get_experience(self):
        if self.hist_len == 1:
            return super(MinisimEnv, self)._get_experience()
        else:
            return Experience(state0=self.exp_state0,  # NOTE: here state0 is always None
                              action=self.exp_action,
                              reward=self.exp_reward,
                              state1=self.state_buffer,
                              terminal1=self.exp_terminal1)

    def reset(self):
        self._reset_experience()
        self.exp_state1 = self.client.reset()
        if self.hist_len > 1:
            self._append_to_history(self._preprocessState(self.exp_state1))
        return self._get_experience()

    def step(self, action_index):
        self.exp_action = action_index
        if self.enable_continuous:
            # TODO: not implemented
            self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.client.step(self.exp_action)
        else:
            # print("actions taken:", [self.actions[i] for i in self._to_n_dim_idx(action_index, self.num_robots)])
            self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.client.step(
                [self.actions[i] for i in self._to_n_dim_idx(action_index, self.num_robots)])
            if self.hist_len > 1:
                self._append_to_history(self._preprocessState(self.exp_state1))
        return self._get_experience()

    def _to_n_dim_idx(self, idx, n_dims):
        res = np.zeros(n_dims, dtype=np.int)
        for i in range(n_dims):
            sub = idx / len(self.actions) ** (n_dims - i - 1)
            if i != n_dims - 1:
                res[i] = sub
                idx -= sub * len(self.actions) ** (n_dims - i - 1)
            else:
                res[i] = idx % len(self.actions)
        return res

    def init_roslaunch(self):
        rospack = roslaunch.rospkg.RosPack()
        try:
            minisim_path = rospack.get_path('minisim')
        except roslaunch.rospkg.ResourceNotFound:
            self.logger.warning("WARNING: minisim not found")
            sys.exit(-1)
        map_server_rlaunch_path = os.path.join(minisim_path, 'launch', 'map_server_small.launch')
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        MinisimEnv.roslaunch_map_server = roslaunch.parent.ROSLaunchParent(uuid, [map_server_rlaunch_path])
        MinisimEnv.roslaunch_map_server.start()

        MinisimEnv.roslaunch_node_starter = roslaunch.scriptapi.ROSLaunch()
        MinisimEnv.roslaunch_node_starter.start()

    def launch_node(self):
        package = 'minisim'
        executable = 'minisim_srv'
        node = roslaunch.core.Node(package, executable, required=True, name=self.sim_name,
                                   namespace=self.sim_name, output='screen')

        return MinisimEnv.roslaunch_node_starter.launch(node)


if __name__ == '__main__':
    env = MinisimEnv(EnvParams(), 0)
    time.sleep(10000)
