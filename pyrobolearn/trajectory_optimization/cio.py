#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This file implements the 'Contact Invariant Optimization' framework developed by Igor Mordatch.

References:
    [1] "Automated Discovery and Learning of Complex Movement Behaviors" (PhD thesis), Mordatch, 2015
    [2] Mordatch's presentation given in CS294
"""

from itertools import count
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from cvxopt import matrix, spmatrix, sparse, spdiag, solvers


class CIO(object):
    r"""
    The Contact Invariant Optimization (CIO) algorithm [1] consists to minimize the following cost:

    .. math::

        s* = argmin_s L(s)
           = argmin_s L_{CI}(s) + L_{physics}(s) + L_{task}(s) + L_{hint}(s)

    where :math:`s` is the state which contains :math:`x_k`, :math:`\dot{x}_k`, and :math:`c_k` for each phase/interval
    :math:`k`. The vector :math:`x_k` contains the torso and end-effector position and orientation, while
    the :math:`c_k` vector represents the auxiliary contact variables.

    Here is what each term represents in the total cost:
    - :math:`L_{CI}` is the contact invariant cost
    - :math:`L_{physics}` penalizes physics violation
    - :math:`L_{task}` describes the task objectives (i.e. high-level goals of the movement)
    - :math:`L_{hint}` provides hints to accelerate the optimization. This term is optional.

    We now describe each cost more specifically.


    * The contact invariant cost is given by:

    .. math:: L_{CI}(s) = \sum_i^N \sum_t^T c_{i, \phi(t)} (||e_{i,t}(s)||^2 + ||\dot{e}_{i,t}(s)||^2)

    where :math:`e_{i,t} = [p_i(q_t) - n'(p), 0]` is the 4D contact-violation vector.

    * The physics violation cost is formulated as:

    .. math:: L_{physics}(s) = \sum_t^T || J_t(s)^T f_t(s) + B u_t(s) - \tau_t(s) ||^2

    where the external contact forcing terms :math:`f_t = [f_1, ..., f_N]^\top \in \mathbb{R}^{6N}` (with each forcing
    term  (for each end-effector) is given by :math:`f_i = [f_c, \tau_c]` where :math:`f_c` is the translational
    contact force, and :math:`\tau_c` is the torsion around the surface normal) and joint actuation
    :math:`u_t \in \mathbb{R}^{D_a}` (where :math:`D_a` are the number of actuated joints and  are computed according to:

    .. math::

        f_t, u_t =& \arg \min_{f, u} || J_t(q_t)^\top f - Bu - \tau_t(q_t, \dot{q}_t, \ddot{q}_t)||^2 + f^\top W_t f +
            u^\top R u \\
            \mbox{subject to } \quad A f \leq b

    which is solved using quadratic programming (QP). The linear constraint is the linear approximation to the friction
    cone (i.e. the friction pyramid). The torques :math:`\tau_t(q_t, \dot{q}_t, \ddot{q}_t)` are given by the whole
    body dynamic equation:

    .. math:: \tau_t(q_t, \dot{q}_t, \ddot{q}_t) = H(q_t) \ddot{q}_t + C(q_t, \dot{q}_t) \dot{q}_t + g(q_t)


    * The task cost is expressed as:

    .. math:: L_{task}(s) = \sum_b l_b(q_T(s)) + \sum_t^T ||f_t(s)||^2 + ||u_t(s)||^2 + ||\ddot{q}_t(s)||^2

    where :math:`b` is an index over different tasks, :math:`l_b` are task specific terms which only depends on the
    final pose :math:`q_T(s)`.


    * The optional hint cost is given by:

    .. math:: L_{hint}(s) = \sum_t \max(||z_t(s) - n(z_t(s))|| - \epsilon, 0)^2

    where :math:`z_t(s) = z_t(q, \ddot{q})` is the zero-moment point (ZMP).


    The CIO consists of 3 phases:
    1. only :math:`L_{task}` is enabled
    2. All 4 terms (:math:`L_{task}`, :math:`L_{physics}`, :math:`L_{CI}`, :math:`L_{hint}`) are enabled but with
       :math:`L_{physics}` down-weighted by 0.1
    3. :math:`L_{task}`, :math:`L_{physics}`, and :math:`L_{CI}` are fully enabled

    Note that the solution obtained at the end of each phase is perturbed with small zero-mean Gaussian noise to
    break any symmetries, and used to initialize the next phase.

    From the optimized state :math:`s^*`, the optimal joints :math:`q^*` at each time step can be computed (using IK
    and cubic spline interpolation). A PD controller can then be used to move the joints to their desired configuration.

    Note that the framework does not take into account any sensory feedback.

    References:
        [1] "Automated Discovery and Learning of Complex Movement Behaviors" (PhD thesis), Mordatch, 2015
    """

    def __init__(self, robot, total_time, interval=0.5, delta_time=0.1):
        # TODO: think about optimizing multiple actors
        self.robot = robot
        self.D = robot.num_dofs
        self.N = robot.num_end_effectors
        self.U = len(robot.actuated_joints)

        self.K = round(total_time / interval)
        self.T = round(total_time / delta_time)
        self.delta_time = delta_time

        x = np.linspace(0, self.T, self.K)
        y = np.array(range(1, self.K + 1))
        self.phase = interp1d(x, y, kind='zero')

        # the values need to be optimized
        self.params = np.concatenate([np.zeros(self.T * self.D), np.ones(self.K * self.N)])
        self._init_joint_positions()

        # the modulating matrix maps actuator space into full space
        self.control_matrix = spmatrix(1.0, robot.actuated_joints, range(self.U), (self.D, self.U))
        self.control_regular = spmatrix(1.0, range(self.U), range(self.U))

    def get_phase_index(self, t):
        return self.phase(t)

    def _init_joint_positions(self):
        base_pos = self.robot.get_base_position()
        base_rot = self.robot.get_base_orientation()
        base_rot = Rotation.from_quat(base_rot).as_rotvec()
        q = np.atleast_2d(np.concatenate([base_pos, base_rot, self.robot.get_joint_positions()]))
        q = q.repeat(self.T, axis=0)

        qs = self.params[:self.T * self.D]
        qs = qs.reshape([self.T, self.D])
        qs[:] = q

    def get_joint_positions(self, t):
        qs = self.params[:self.T * self.D]
        qs = qs.reshape([self.T, self.D])
        return qs[t]

    def get_contacts(self, t):
        cs = self.params[self.T * self.D:]
        cs = cs.reshape([self.K, self.N])
        index = int(self.get_phase_index(t).tolist())
        return cs[index]

    def _task_cost(self):
        pass

    def _contact_invariant_cost(self):
        pass

    def physics_cost(self):
        cost = 0

        for t in range(1, self.T - 1):
            # do finite differences
            q = self.get_joint_positions(t)
            dq = (self.get_joint_positions(t + 1) - self.get_joint_positions(t - 1)) / (2 * self.delta_time)
            ddq = (self.get_joint_positions(t + 1) - 2 * q + self.get_joint_positions(t - 1)) / (self.delta_time ** 2)

            # solve inverse dynamics for tau
            base_pos = q[:3]
            base_rot = q[3:6]
            joint_pos = q[6:]

            base_rot = Rotation.from_rotvec(base_rot).as_quat()
            tau = self.robot.calculate_inverse_dynamics(ddq, dq, np.concatenate([base_pos, base_rot, joint_pos]))
            tau = matrix(tau)

            force_regular = (0.01 / (self.get_contacts(t) ** 2 + 0.001)).repeat(6)
            force_regular = spdiag(force_regular.tolist())
            regular = spdiag([force_regular, self.control_regular])

            jac = np.vstack([robot.get_jacobian(i, joint_pos) for i in robot.end_effectors])
            jac = sparse(jac.tolist())
            combined = sparse([jac.T, self.control_matrix.T])

            second_order = 2 * combined * combined.T + 2 * regular
            first_order = - combined * tau
            solution = solvers.qp(second_order, first_order)

            cost += sum((combined.T * solution['x'] - tau) ** 2)

        return cost

    def optimize(self):
        pass


if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import Bullet
    from pyrobolearn.worlds import BasicWorld
    from pyrobolearn.robots import ANYmal

    sim = Bullet()
    world = BasicWorld(sim)

    robot = ANYmal(sim)
    # robot.disable_motor()

    robot.print_info()

    world.load_robot(robot)

    cio = CIO(robot, 10)
    cio.physics_cost()

    for i in count():
        q = np.concatenate([robot.get_base_pose(concatenate=True), robot.get_joint_positions()])
        dq = np.concatenate([robot.get_base_velocity(), robot.get_joint_velocities()])
        ddq = np.zeros_like(dq)

        tau = robot.calculate_inverse_dynamics(ddq, dq, q)
        # robot.set_joint_torques(tau[6:])

        link_id = robot.get_end_effector_ids(0)
        q = robot.get_joint_positions()
        J = robot.get_jacobian(link_id, q)

        world.step(sim.dt)
