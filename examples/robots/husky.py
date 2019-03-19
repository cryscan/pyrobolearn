#!/usr/bin/env python
"""Load the Husky robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Husky

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Husky(sim)

# print information about the robot
robot.printRobotInfo()

# Position control using sliders
# robot.addJointSlider()

# run simulator
for _ in count():
    # robot.updateJointSlider()
    robot.driveForward(2)
    world.step(sleep_dt=1./240)
