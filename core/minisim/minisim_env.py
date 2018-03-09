from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import roslaunch
import rosparam
import subprocess
import imageio

from core.env import Env
from core.minisim.minisim_client import MinisimClient
from utils.helpers import Experience
from utils.options import EnvParams


# TODO: figure out logging
class MinisimEnv(Env):
    initialized = False
    minisim_path = None
    roslaunch_map_server = None
    roslaunch_node_starter = None
    roscore = None
    map_dir = None

    def __init__(self, args, env_ind=0):
        tmp = self._reset_experience
        self._reset_experience = lambda: None
        super(MinisimEnv, self).__init__(args, env_ind)
        self._reset_experience = tmp

        assert self.env_type == "minisim"

        self.extras = None

        self.num_robots = args.num_robots
        self.curriculum = args.curriculum if hasattr(args, "curriculum") else False
        self.randomize_maps = args.randomize_maps if hasattr(args, "randomize_maps") else False
        self.randomize_targets = args.randomize_targets if hasattr(args, "randomize_targets") else False
        self.penalize_staying = args.penalize_staying if hasattr(args, "penalize_staying") else False
        self.penalize_angle_to_target = args.penalize_angle_to_target if hasattr(args,
                                                                                 "penalize_angle_to_target") else False
        self.collision_is_terminal = args.collision_is_terminal if hasattr(args, "collision_is_terminal") else False

        self.mode = args.mode  # 1(train) | 2(test model_file)
        self.total_reward = 0

        if self.mode == 2:
            self.curriculum = False
            self.collision_is_terminal = False

        self.sim_name = 'sim' + str(self.ind)

        if not MinisimEnv.initialized:
            self._init_roslaunch()

        self.node = self._launch_node()
        self.client = MinisimClient(
            self.num_robots, self.seed, self.curriculum, self.mode,
            self.randomize_targets, self.penalize_staying,
            self.penalize_angle_to_target, self.collision_is_terminal,
            '/' + self.sim_name, self.logger
        )
        self.client.setup()  # TODO: move to client's init?

        # action space setup  # [linear velocity, angular velocity]
        # seemed to be too small
        # self.actions = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]  # ,[1, 1], [1, -1], [-1, 1], [-1, -1]]
        # definitely too huge, only realised a few months in :(
        # self.actions = [[0, 0], [10, 0], [-10, 0], [0, 16], [0, -16]]  # ,[1, 1], [1, -1], [-1, 1], [-1, -1]]
        # trying out
        # self.actions = [[0, 0], [3, 0], [-3, 0], [0, 8], [0, -8]]  # ,[1, 1], [1, -1], [-1, 1], [-1, -1]]
        # trying out without the option to stand still and backwards movement
        self.actions = [[3, 0], [0, 8], [0, -8]]
        # try to promote more realistic behavior with slower backward movement?
        # self.actions = [[0, 0], [3, 0], [-1, 0], [0, 8], [0, -8]]  # ,[1, 1], [1, -1], [-1, 1], [-1, -1]]
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

        # TODO: history is currently broken (however it was not useful according to the experiments anyway)
        # it was harmful, even
        if hasattr(args, "hist_len"):
            self.hist_len = args.hist_len
            self.state_buffer = np.zeros((self.hist_len, self.state_shape + 2 * self.num_robots))
        else:
            self.hist_len = 1

        self._reset_experience()

    def __del__(self):
        if self.node is not None:
            self.node.stop()

    def _preprocessState(self, state):
        return state

    def _reset_experience(self):
        super(MinisimEnv, self)._reset_experience()
        self.extras = None
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
        return len(self.actions)

    def render(self):
        self.logger.warning("WARNING: asked to render minisim - user rviz instead")

    def visual(self):
        pass

    def sample_random_action(self):  # TODO: unused
        return [self.actions[np.random.randint(0, len(self.actions))] for _ in xrange(self.num_robots)]

    def _get_experience(self):
        if self.hist_len == 1:
            return Experience(state0=self.exp_state0,  # NOTE: here state0 is always None
                              action=self.exp_action,
                              reward=self.exp_reward,
                              state1=self._preprocessState(self.exp_state1),
                              terminal1=self.exp_terminal1,
                              extras=self.extras)
        else:
            return Experience(state0=self.exp_state0,  # NOTE: here state0 is always None
                              action=self.exp_action,
                              reward=self.exp_reward,
                              state1=self.state_buffer,
                              terminal1=self.exp_terminal1,
                              extras=self.extras)

    def reset(self):
        self._reset_experience()
        self.exp_state1, self.extras = self.client.reset()
        if self.hist_len > 1:
            self._append_to_history(self._preprocessState(self.exp_state1))
        self.total_reward = 0
        return self._get_experience()

    def step(self, action_index):
        self.exp_action = action_index
        if self.enable_continuous:
            # TODO: not implemented
            self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.client.step(self.exp_action)
        else:
            # enumerated action combinations
            # print("actions taken:", [self.actions[i] for i in self._to_n_dim_idx(action_index, self.num_robots)])
            # self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.client.step(
            #     [self.actions[i] for i in self._to_n_dim_idx(action_index, self.num_robots)]
            # )

            # unstructured reward
            # self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.client.step(
            #     [self.actions[i] for i in action_index.reshape(-1)]
            # )

            # structured reward
            self.exp_state1, self.exp_reward, self.exp_terminal1, self.extras, _ = self.client.step_structured(
                [self.actions[i] for i in action_index.reshape(-1)]
            )
            if self.mode == 2:
                # time.sleep(0.33)
                self.total_reward += self.exp_reward
                print('total reward: ', self.total_reward)
                # print("actions: ", action_index)
            if self.hist_len > 1:
                self._append_to_history(self._preprocessState(self.exp_state1))
        return self._get_experience()

    def read_static_map_image(self):
        # return imageio.imread(os.path.join(MinisimEnv.minisim_path, 'map', 'medium_rooms.pgm'))
        # return imageio.imread(os.path.join(MinisimEnv.minisim_path,
        #                                    'map', 'random', 'simple_gen_small_002.pgm'))
        return imageio.imread(os.path.join(MinisimEnv.minisim_path,
                                           'map', 'medium_rooms_simpler.pgm'))
        # return imageio.imread(os.path.join(MinisimEnv.minisim_path,
        #                                    'map', 'medium_rooms_new.pgm'))

    # was supposed to be useful for a large network with a single action index output, which would
    # be expanded into individual robot actions
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

    def _init_roslaunch(self):
        rospack = roslaunch.rospkg.RosPack()
        try:
            minisim_path = rospack.get_path('minisim')
            MinisimEnv.minisim_path = minisim_path
        except roslaunch.rospkg.ResourceNotFound:
            self.logger.warning("WARNING: minisim not found")
            sys.exit(-1)

        if not self.randomize_maps:
            # TODO: find a way to provide the map file arg to the map_server launch file
            # map_server_rlaunch_path = os.path.join(minisim_path, 'launch', 'map_server_small.launch')
            # map_server_rlaunch_path = os.path.join(minisim_path, 'launch', 'map_server_small_simple.launch')
            # map_server_rlaunch_path = os.path.join(minisim_path, 'launch', 'map_server_empty_small.launch')
            # map_server_rlaunch_path = os.path.join(minisim_path, 'launch', 'map_server_simple_gen_small_002.launch')
            # map_server_rlaunch_path = os.path.join(minisim_path, 'launch', 'map_server_medium_rooms.launch')
            map_server_rlaunch_path = os.path.join(minisim_path, 'launch', 'map_server_medium_rooms_simpler.launch')
            # map_server_rlaunch_path = os.path.join(minisim_path, 'launch', 'map_server_medium_rooms_new.launch')
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            MinisimEnv.roslaunch_map_server = roslaunch.parent.ROSLaunchParent(uuid, [map_server_rlaunch_path])
            MinisimEnv.roslaunch_map_server.start()
        else:
            master = roslaunch.scriptapi.Master()
            if not master.is_running():
                MinisimEnv.roscore = subprocess.Popen('roscore')

            rlaunch_path = os.path.join(minisim_path, 'launch', 'sim_srv_multimap.launch')
            loader = roslaunch.xmlloader.XmlLoader(resolve_anon=False)
            config = roslaunch.config.ROSLaunchConfig()
            loader.load(rlaunch_path, config, verbose=False)
            MinisimEnv.map_dir = config.params.values()[0].value

        MinisimEnv.roslaunch_node_starter = roslaunch.scriptapi.ROSLaunch()
        MinisimEnv.roslaunch_node_starter.start()
        MinisimEnv.initialized = True

    def _launch_node(self):
        package = 'minisim'
        executable = 'minisim_srv' if not self.randomize_maps else 'minisim_srv_standalone'
        node = roslaunch.core.Node(package, executable, required=True, name=self.sim_name,
                                   namespace=self.sim_name, output='screen')

        if self.randomize_maps:
            rosparam.set_param("/{0}/{0}/map_dir".format(self.sim_name), MinisimEnv.map_dir)

        return MinisimEnv.roslaunch_node_starter.launch(node)


if __name__ == '__main__':
    params = EnvParams()

    env_0 = MinisimEnv(params, 0)
    env_1 = MinisimEnv(params, 1)
    time.sleep(10000)
