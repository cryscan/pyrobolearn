#!/usr/bin/env python
"""Load the Jaco robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Jaco

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Jaco(sim)

# print information about the robot
robot.printRobotInfo()

# run simulation
for i in count():
    # step in simulation
    world.step(sleep_dt=1./240)
