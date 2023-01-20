import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from casadi import *
import casadi.tools as ct


class MPC:
    def __init__(self, K=2, N=30, Q=1e-3, R=1e-4, dt=0.02):
        self.mpc_solver = None
        self.K = K #2
        self.N = N #30
        self.Q = Q #20
        self.R = R #10

        self.dt = dt

        self.lb_opt_x = None
        self.ub_opt_x = None
        self.opt_x = None

        self.max_x = 3.0 #8.0
        self.max_u = 30 #2.0

    def generate_solver(self):
        ## System declaration
        # some common parameters
        # Timestep
        self.dt = 0.02

        # Dimensions of x and u:
        nx = 4
        nu = 1

        # physical parameters
        g = -9.81
        mc = 2.4
        mp = 0.23
        lp = 0.36
        # initial state
        x_0 = np.array([0.5, 0, 3.1, 0]).reshape(nx, 1)

        # symbolic declaration
        x = SX.sym("x", nx, 1)
        u = SX.sym("u", nu, 1)

        # System dynamics
        xdot = vertcat(
            x[1],
            (u + mp * lp * sin(x[2]) * x[3] ** 2 - mp * g * cos(x[2]) * sin(x[2]))
            / (mc + mp - mp * cos(x[2]) ** 2),
            x[3],
            (u * cos(x[2]) + (mc + mp) * g * sin(x[2]) + mp * lp * cos(x[2]) * sin(x[2]) * x[3] ** 2)
            / (mp * lp * cos(x[2]) ** 2 - (mc + lp) * lp),
        )

        system = Function("system", [x, u], [xdot])

        ## Collocation settings
        # collocation degree
        K = self.K
        # collocation points (excluding 0)
        tau_col = collocation_points(K, 'radau')
        # collocation points (including 0)
        tau_col = [0] + tau_col

        ## Lagrange Polynomials generation
        def L(tau_col, tau, j):
            l = 1
            for k in range(len(tau_col)):
                if k != j:
                    l *= (tau - tau_col[k]) / (tau_col[j] - tau_col[k])
            return l

        ## Calculate orthogonal collocation points
        tau = SX.sym('tau')
        A = np.zeros((K + 1, K + 1))
        for j in range(K + 1):
            dLj = gradient(L(tau_col, tau, j), tau)
            dLj_fcn = Function('dLj_fcn', [tau], [dLj])
            for k in range(K + 1):
                A[j, k] = dLj_fcn(tau_col[k])

        ## Calculate continuity coefficients
        D = np.zeros((K + 1, 1))
        for j in range(K + 1):
            Lj = L(tau_col, tau, j)
            Lj_fcn = Function('Lj', [tau], [Lj])
            D[j] = Lj_fcn(1)

        ## MPC with Orthogonal Collocation
        Q = self.Q
        Q = Q * np.diag(np.ones(nx))
        #Q[0, 0] *= 1.0
        #Q[1, 1] *= 1.0

        R = self.R
        R = np.diag(R * np.ones(nu))

        # stage cost
        stage_cost = x.T @ Q @ x + u.T @ R @ u
        stage_cost_fcn = Function('stage_cost', [x, u], [stage_cost])

        # terminal cost
        terminal_cost = x.T @ Q @ x
        terminal_cost_fcn = Function('terminal_cost', [x], [terminal_cost])

        # Prediction Horizon
        N = self.N

        # state constraints
        lb_x = -self.max_x * np.ones((nx, 1))
        ub_x = self.max_x * np.ones((nx, 1))
        # input constraints
        lb_u = - self.max_u * np.ones((nu, 1))
        ub_u = self.max_u * np.ones((nu, 1))

        # Define Optimization variable
        self.opt_x = ct.struct_symSX([
            ct.entry('x', shape=nx, repeat=[N + 1, K + 1]),
            ct.entry('u', shape=nu, repeat=[N])
        ])

        # Upper & lower bounds
        self.lb_opt_x = self.opt_x(0)
        self.ub_opt_x = self.opt_x(0)
        self.lb_opt_x['x'] = lb_x
        self.ub_opt_x['x'] = ub_x
        self.lb_opt_x['u'] = lb_u
        self.ub_opt_x['u'] = ub_u

        self.lb_g = None
        self.ub_g = None

        ## formulate MPC optimization problem
        # initialize empty variables
        J = 0
        g = []  # constraint expression g
        self.lb_g = []  # lower bound for constraint expression g
        self.ub_g = []  # upper bound for constraint expression g

        # 01 - Your code here!
        x_init = SX.sym('x_init', nx)

        x0 = self.opt_x['x', 0, 0]

        g.append(x0 - x_init)
        self.lb_g.append(np.zeros((nx, 1)))
        self.ub_g.append(np.zeros((nx, 1)))
        # 01

        for i in range(N):
            # 02 - Your code here!
            # objective
            J += stage_cost_fcn(self.opt_x['x', i, 0], self.opt_x['u', i])
            # 02
            # 03 - Your code here!
            # equality constraints (system equation)
            for k in range(1, K + 1):
                gk = -self.dt * system(self.opt_x['x', i, k], self.opt_x['u', i])
                for j in range(K + 1):
                    gk += A[j, k] * self.opt_x['x', i, j]

                g.append(gk)
                self.lb_g.append(np.zeros((nx, 1)))
                self.ub_g.append(np.zeros((nx, 1)))
            # 03
            # 04 - Your code here!
            x_next = horzcat(*self.opt_x['x', i]) @ D
            g.append(x_next - self.opt_x['x', i + 1, 0])
            self.lb_g.append(np.zeros((nx, 1)))
            self.ub_g.append(np.zeros((nx, 1)))
            # 04
        # 05 - Your code here!
        J += terminal_cost_fcn(self.opt_x['x', N, 0])
        # 05
        # 06 - Your code here!
        g = vertcat(*g)
        lb_g = vertcat(*self.lb_g)
        ub_g = vertcat(*self.ub_g)

        prob = {'f': J, 'x': vertcat(self.opt_x), 'g': g, 'p': x_init}
        self.mpc_solver = nlpsol('solver', 'ipopt', prob)
        # 06

        return self.mpc_solver

