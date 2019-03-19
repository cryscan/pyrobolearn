#!/usr/bin/env python
"""Load the Kuka LWR robotic platform.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaLWR

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = KukaLWR(sim)

# print information about the robot
robot.printRobotInfo()
# H = robot.calculateMassMatrix()
# print("Inertia matrix: H(q) = {}".format(H))

for i in count():
    # step in simulation
    world.step(sleep_dt=1./240)
