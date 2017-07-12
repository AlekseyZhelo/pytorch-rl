from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import geometry_msgs.msg
import numpy as np
import rospy
import time
from minisim.srv import *


class MinisimClient(object):
    laser_rays_per_robot = 361
    message_ok = "ok"

    def __init__(self, num_robots, seed, sim_prefix, logger):
        self.num_robots = num_robots
        self.seed = seed
        self.sim_prefix = sim_prefix
        self.logger = logger
        self.twists = [geometry_msgs.msg.Twist() for _ in xrange(self.num_robots)]

        self.setup_robots_name = self.sim_prefix + '/setup_robots'
        self.simulation_step_name = self.sim_prefix + '/simulation_step'
        self.reset_simulation_name = self.sim_prefix + '/reset_simulation'
        rospy.wait_for_service(self.setup_robots_name)
        rospy.wait_for_service(self.simulation_step_name)
        rospy.wait_for_service(self.reset_simulation_name)

        self.setup_robots = rospy.ServiceProxy(self.setup_robots_name, SetupRobots, persistent=True)
        self.simulation_step = rospy.ServiceProxy(self.simulation_step_name, SimulationStep, persistent=True)
        self.reset_simulation = rospy.ServiceProxy(self.reset_simulation_name, ResetSimulation, persistent=True)

    @property
    def state_shape(self):
        return self.num_robots * MinisimClient.laser_rays_per_robot

    def setup(self):
        try:
            resp = self.setup_robots(self.num_robots, self.seed)
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
                return np.array(resp.state), resp.reward, resp.terminal, resp.message
        except rospy.ServiceException, e:
            print("SimulationStep service call failed: %s" % e)

    def reset(self):
        try:
            resp = self.reset_simulation()
            if resp.message != MinisimClient.message_ok:
                self.logger.warning(
                    "WARNING: ResetSimulation service improperly called, message: {0}".format(resp.message))
                return None
            else:
                return np.array(resp.state)
        except rospy.ServiceException, e:
            print("ResetSimulation service call failed: %s" % e)


if __name__ == '__main__':
    class DummyLogger(object):
        def warning(self, *args, **kwargs):
            print(args, kwargs)


    client = MinisimClient(1, 213, "", DummyLogger())
    client.setup()
    # print(client.reset())
    actions = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

    steps = 10000
    start = time.time()
    for i in xrange(steps):
        if i % 1000 == 0:
            client.reset()
        client.step([actions[np.random.randint(0, len(actions))]])
    print ("Time elapsed for {0} steps: {1} sec".format(steps, time.time() - start))
    # about 11 seconds for 10000 steps, seems too slow
    # and about 9 seconds without resets every 1000 steps, why do they take so much time?
