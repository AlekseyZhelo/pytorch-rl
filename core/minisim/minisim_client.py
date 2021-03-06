from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import geometry_msgs.msg
import numpy as np
import rospy
import time
from minisim.srv import *
from datetime import datetime


class MinisimClient(object):
    laser_rays_per_robot = 72
    message_ok = "ok"

    def __init__(self, num_robots, seed, curriculum, mode, randomize_targets, penalize_staying,
                 penalize_angle_to_target, collision_is_terminal, sim_prefix, logger):
        self.num_robots = num_robots
        self.seed = seed
        self.curriculum = curriculum
        self.mode = mode  # 1(train) | 2(test model_file)
        self.randomize_targets = randomize_targets
        self.penalize_staying = penalize_staying
        self.penalize_angle_to_target = penalize_angle_to_target
        self.collision_is_terminal = collision_is_terminal
        self.sim_prefix = sim_prefix
        self.logger = logger
        self.twists = [geometry_msgs.msg.Twist() for _ in xrange(self.num_robots)]

        self.setup_robots_name = self.sim_prefix + '/setup_robots'
        self.simulation_step_name = self.sim_prefix + '/simulation_step'
        self.simulation_step_structured_name = self.sim_prefix + '/simulation_step_structured'
        self.reset_simulation_name = self.sim_prefix + '/reset_simulation'
        rospy.wait_for_service(self.setup_robots_name)
        rospy.wait_for_service(self.simulation_step_name)
        rospy.wait_for_service(self.simulation_step_structured_name)
        rospy.wait_for_service(self.reset_simulation_name)

        self.setup_robots = rospy.ServiceProxy(self.setup_robots_name, SetupRobots, persistent=True)
        self.simulation_step = rospy.ServiceProxy(self.simulation_step_name, SimulationStep, persistent=True)
        self.simulation_step_structured = rospy.ServiceProxy(self.simulation_step_structured_name,
                                                             SimulationStepStructured, persistent=True)
        self.reset_simulation = rospy.ServiceProxy(self.reset_simulation_name, ResetSimulation, persistent=True)

        # self.debug_start_time = datetime.now()
        # self.debug_episode_log = open('episode_log_{0}.txt'.format(self.debug_start_time.strftime('%Y-%m-%d_%H_%M_%S')), 'w')

    @property
    def state_shape(self):
        # return self.num_robots * MinisimClient.laser_rays_per_robot
        return MinisimClient.laser_rays_per_robot  # batch, not all together as one input

    def setup(self):
        try:
            resp = self.setup_robots(
                self.num_robots, self.seed, self.curriculum, self.mode,
                self.randomize_targets, self.penalize_staying, self.penalize_angle_to_target,
                self.collision_is_terminal
            )
            if resp.message != MinisimClient.message_ok:
                self.logger.warning("WARNING: SetupRobots service improperly called, message: {0}".format(resp.message))
        except rospy.ServiceException, e:
            print("SetupRobots service call failed: %s" % e)

    def step(self, actions):
        if len(actions) != self.num_robots:
            self.logger.warning(
                "WARNING: Wrong number of actions in SimulationStep: expected {0}, received {1}".format(self.num_robots,
                                                                                                        len(actions)))
            return None

        for i, vel in enumerate(actions):
            self.twists[i].linear.x = vel[0]
            self.twists[i].angular.z = vel[1]

        try:
            resp = self.simulation_step(self.twists)
            if resp.message != MinisimClient.message_ok:
                self.logger.warning(
                    "WARNING: SimulationStep service improperly called, message: {0}".format(resp.message))
                return None
            else:
                # return np.array(resp.state), resp.reward, resp.terminal, resp.message
                return np.array(resp.state).reshape(self.num_robots, -1), resp.reward, resp.terminal, resp.message
        except rospy.ServiceException, e:
            print("SimulationStep service call failed: %s" % e)

    def step_structured(self, actions):
        if len(actions) != self.num_robots:
            self.logger.warning(
                "WARNING: Wrong number of actions in SimulationStepStructured: expected {0}, received {1}".format(
                    self.num_robots,
                    len(actions)))
            return None

        for i, vel in enumerate(actions):
            self.twists[i].linear.x = vel[0]
            self.twists[i].angular.z = vel[1]

        try:
            resp = self.simulation_step_structured(self.twists)
            if resp.message != MinisimClient.message_ok:
                self.logger.warning(
                    "WARNING: SimulationStepStructured service improperly called, message: {0}".format(resp.message))
                return None
            else:
                # return np.array(resp.state), resp.reward, resp.terminal, resp.message
                return np.array(resp.state).reshape(self.num_robots, -1), np.array(
                    resp.reward), resp.terminal, dict(robot_map_x=resp.robot_map_x,
                                                      robot_map_y=resp.robot_map_y,
                                                      robot_theta=resp.robot_theta), resp.message
        except rospy.ServiceException, e:
            print("SimulationStepStructured service call failed: %s" % e)

    def reset(self):
        try:
            resp = self.reset_simulation()
            if resp.message != MinisimClient.message_ok:
                self.logger.warning(
                    "WARNING: ResetSimulation service improperly called, message: {0}".format(resp.message))
                return None
            else:
                # print(resp.robot_map_x, resp.robot_map_y, resp.robot_theta, resp.target_map_x, resp.target_map_y,
                #       file=self.debug_episode_log)
                return np.array(resp.state).reshape(self.num_robots, -1), \
                       dict(
                           robot_map_x=resp.robot_map_x,
                           robot_map_y=resp.robot_map_y,
                           robot_theta=resp.robot_theta,
                           target_map_x=resp.target_map_x,
                           target_map_y=resp.target_map_y,
                           target_radius=resp.target_radius
                       )
        except rospy.ServiceException, e:
            print("ResetSimulation service call failed: %s" % e)


if __name__ == '__main__':
    class DummyLogger(object):
        def warning(self, *args, **kwargs):
            print(*args, file=sys.stderr, **kwargs)


    client = MinisimClient(num_robots=1, seed=213, curriculum=False, mode=1,
                           randomize_targets=True, penalize_staying=True,
                           penalize_angle_to_target=True, collision_is_terminal=True,
                           sim_prefix="", logger=DummyLogger())
    client.setup()
    # print(client.reset())
    actions = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

    steps = 10000
    start = time.time()
    for i in xrange(steps):
        if i % 1000 == 0:
            client.reset()
        client.step([actions[np.random.randint(0, len(actions))]])
    print("Time elapsed for {0} steps: {1} sec".format(steps, time.time() - start))
    # about 11 seconds for 10000 steps, seems too slow
    # and about 9 seconds without resets every 1000 steps, why do they take so much time?
