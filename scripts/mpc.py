import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from casadi import *
import casadi.tools as ct

import cartpole_sim
from visualization import plot_trajectories, animate_system


class MPC:
    def __init__(
        self,
        K=2,
        N=30,
        Q=np.array(
            [
                [1e-2, 0.0, 0.0, 0.0],
                [0.0, 1e-2, 0.0, 0.0],
                [0.0, 0.0, 1e-2, 0.0],
                [0.0, 0.0, 0.0, 1e-2],
            ]
        ),
        Qf=np.array(
            [
                [1e-2, 0.0, 0.0, 0.0],
                [0.0, 1e-2, 0.0, 0.0],
                [0.0, 0.0, 1e-2, 0.0],
                [0.0, 0.0, 0.0, 1e-2],
            ]
        ),
        R=1e-4,
        dt=0.02,
        x_bound=4.0,
        xdot_bound=50.0,
        theta_bound=4.0,
        thetadot_bound=50.0,
        u_bound=25.0,
        max_solver_iter=20,
    ):
        self.mpc_solver = None
        self.K = K  # 2
        self.N = N  # 30
        self.Q = Q  # 20
        self.Qf = Qf
        self.R = R  # 10

        self.nx = 4
        self.nu = 1

        self.setpoint = np.zeros([self.nx, 1], dtype=np.float64)

        self.dt = dt
        self.max_solver_iter =max_solver_iter

        self.lb_opt_x = None
        self.ub_opt_x = None
        self.opt_x = None

        self.lb_g = None
        self.ub_g = None

        self.max_x = x_bound
        self.max_xdot = xdot_bound
        self.max_theta = theta_bound
        self.max_thetadot = thetadot_bound
        self.max_u = u_bound

    def generate(self):
        ## System declaration
        # Dimensions of x and u:
        nx = self.nx
        nu = self.nu

        # physical parameters
        g = -9.81
        mc = 2.4
        mp = 0.23
        lp = 0.36

        # initial state
        # x_0 = self.x_0

        # symbolic declaration
        x = SX.sym("x", nx, 1)
        s = SX.sym("s", nx, 1)  # setpoint
        u = SX.sym("u", nu, 1)

        # System dynamics
        xdot = vertcat(
            x[1],
            (u + mp * lp * sin(x[2]) * x[3] ** 2 - mp * g * cos(x[2]) * sin(x[2]))
            / (mc + mp - mp * cos(x[2]) ** 2),
            x[3],
            (
                u * cos(x[2])
                + (mc + mp) * g * sin(x[2])
                + mp * lp * cos(x[2]) * sin(x[2]) * x[3] ** 2
            )
            / (mp * lp * cos(x[2]) ** 2 - (mc + lp) * lp),
        )

        system = Function("system", [x, u], [xdot])

        ## Collocation settings
        # collocation degree
        K = self.K
        # collocation points (excluding 0)
        tau_col = collocation_points(K, "radau")
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
        tau = SX.sym("tau")
        A = np.zeros((K + 1, K + 1))
        for j in range(K + 1):
            dLj = gradient(L(tau_col, tau, j), tau)
            dLj_fcn = Function("dLj_fcn", [tau], [dLj])
            for k in range(K + 1):
                A[j, k] = dLj_fcn(tau_col[k])

        ## Calculate continuity coefficients
        D = np.zeros((K + 1, 1))
        for j in range(K + 1):
            Lj = L(tau_col, tau, j)
            Lj_fcn = Function("Lj", [tau], [Lj])
            D[j] = Lj_fcn(1)

        ## MPC with Orthogonal Collocation
        Q = self.Q
        Qf = self.Qf

        R = self.R
        R = np.diag(R * np.ones(nu))

        # stage cost
        stage_cost = (x - s).T @ Q @ (x - s) + u.T @ R @ u
        stage_cost_fcn = Function("stage_cost", [x, s, u], [stage_cost])

        # terminal cost
        terminal_cost = (x - s).T @ Qf @ (x - s)
        terminal_cost_fcn = Function("terminal_cost", [x, s], [terminal_cost])

        # Prediction Horizon
        N = self.N

        # state constraints
        lb_x = np.empty((nx, 1))
        ub_x = lb_x.copy()
        lb_x[0] = -self.max_x
        ub_x[0] = self.max_x
        lb_x[1] = -self.max_xdot
        ub_x[1] = self.max_xdot
        lb_x[2] = -self.max_theta
        ub_x[2] = self.max_theta
        lb_x[3] = -self.max_thetadot
        ub_x[3] = self.max_thetadot

        # set point constraints
        lb_s = np.empty((nx, 1))
        ub_s = lb_s.copy()
        lb_s[0] = self.setpoint[0]
        ub_s[0] = self.setpoint[0]
        lb_s[1] = self.setpoint[1]
        ub_s[1] = self.setpoint[1]
        lb_s[2] = self.setpoint[2]
        ub_s[2] = self.setpoint[2]
        lb_s[3] = self.setpoint[3]
        ub_s[3] = self.setpoint[3]

        # input constraints
        lb_u = -self.max_u * np.ones((nu, 1))
        ub_u = self.max_u * np.ones((nu, 1))

        # Define Optimization variable
        self.opt_x = ct.struct_symSX(
            [ct.entry("x", shape=nx, repeat=[N + 1, K + 1]),
             ct.entry("s", shape=nx, repeat=[N + 1, K + 1]),
             ct.entry("u", shape=nu, repeat=[N])]
        )

        # Upper & lower bounds
        self.lb_opt_x = self.opt_x(0)
        self.ub_opt_x = self.opt_x(0)
        self.lb_opt_x["x"] = lb_x
        self.ub_opt_x["x"] = ub_x
        self.lb_opt_x["s"] = lb_s
        self.ub_opt_x["s"] = ub_s
        self.lb_opt_x["u"] = lb_u
        self.ub_opt_x["u"] = ub_u

        self.lb_g = None
        self.ub_g = None

        ## formulate MPC optimization problem
        # initialize empty variables
        J = 0
        g = []  # constraint expression g
        self.lb_g = []  # lower bound for constraint expression g
        self.ub_g = []  # upper bound for constraint expression g

        # 01 - Your code here!
        x_init = SX.sym("x_init", nx)

        x0 = self.opt_x["x", 0, 0]

        g.append(x0 - x_init)
        self.lb_g.append(np.zeros((nx, 1)))
        self.ub_g.append(np.zeros((nx, 1)))
        # 01

        for i in range(N):
            # 02 - Your code here!
            # objective
            J += stage_cost_fcn(self.opt_x["x", i, 0], self.opt_x["s", i, 0], self.opt_x["u", i])
            # 02
            # 03 - Your code here!
            # equality constraints (system equation)
            for k in range(1, K + 1):
                gk = -self.dt * system(self.opt_x["x", i, k], self.opt_x["u", i])
                for j in range(K + 1):
                    gk += A[j, k] * self.opt_x["x", i, j]

                g.append(gk)
                self.lb_g.append(np.zeros((nx, 1)))
                self.ub_g.append(np.zeros((nx, 1)))
            # 03
            # 04 - Your code here!
            x_next = horzcat(*self.opt_x["x", i]) @ D
            g.append(x_next - self.opt_x["x", i + 1, 0])
            self.lb_g.append(np.zeros((nx, 1)))
            self.ub_g.append(np.zeros((nx, 1)))
            # 04
        # 05 - Your code here!
        J += terminal_cost_fcn(self.opt_x["x", N, 0], self.opt_x["s", N, 0])
        # 05
        # 06 - Your code here!
        g = vertcat(*g)
        self.lb_g = vertcat(*self.lb_g)
        self.ub_g = vertcat(*self.ub_g)

        prob = {"f": J, "x": vertcat(self.opt_x), "g": g, "p": x_init}
        opts = {"ipopt.print_level": 2, "print_time": 0, "ipopt.max_iter": int(self.max_solver_iter)}
        self.controller = nlpsol("solver", "ipopt", prob, opts)

    def update_state(self, current_state):
        for i in range(4):
            self.lb_opt_x["x", 0, 0, i] = current_state[i]
            self.ub_opt_x["x", 0, 0, i] = current_state[i]


def simulate(params_dict, x_0=np.array([0.5, 0, 0.0, 0]).reshape([-1, 1])):
    # Configure controller
    solver = MPC(
        K=params_dict["MPC_K"],
        N=params_dict["MPC_N"],
        Q=np.diag(
            [
                params_dict["MPC_Q_0"],
                params_dict["MPC_Q_1"],
                params_dict["MPC_Q_2"],
                params_dict["MPC_Q_3"],
            ]
        ),
        R=params_dict["MPC_R"],
        dt=params_dict["dt"],
        x_bound=params_dict["MPC_x_bound"],
        xdot_bound=params_dict["MPC_xdot_bound"],
        theta_bound=params_dict["MPC_theta_bound"],
        thetadot_bound=params_dict["MPC_thetadot_bound"],
        u_bound=params_dict["MPC_u_bound"],
        max_solver_iter=params_dict["max_solver_iter"],
    )
    solver.setpoint[0] = -5
    solver.generate()

    # Configure simulator
    pendulum = cartpole_sim.PendulumOnCart(
        initial_states=x_0[:4], dt=params_dict["dt"], render=True
    )

    # Loop Variables
    # initialize result lists for states and inputs
    res_x_mpc = [x_0[:4]]
    res_x_mpc_full = [x_0]
    res_u_mpc = []

    # set number of iterations
    N_time = params_dict["N_time"]
    N_sim = int(N_time / params_dict["dt"])
    u_k = 0
    # simulation loop
    i = 0
    sgn = -1
    x_next = x_0[:4]
    for k in range(N_sim):
        # solve optimization problem
        i += 1
        if i > 150:
            solver.setpoint[0] += sgn * 10
            print("regenerating")
            solver.generate()
            sgn *= -1
            i = 0

        solver.update_state(x_next)
        mpc_res = solver.controller(p=x_next, lbg=0, ubg=0, lbx=solver.lb_opt_x, ubx=solver.ub_opt_x)

        # Extract the control input
        opt_x_k = solver.opt_x(mpc_res["x"])
        u_k = opt_x_k["u", 0]

        # simulate the system
        x_next = pendulum.step(action=u_k)

        # Update the initial state
        x_0[:4] = x_next

        # Store the results
        res_x_mpc.append(x_next)
        res_x_mpc_full.append(np.concatenate((x_next, x_0[-2:])))
        res_u_mpc.append(u_k)

    # Make an array from the list of arrays:
    res_x_mpc = np.concatenate(res_x_mpc, axis=1)
    res_x_mpc_full = np.concatenate(res_x_mpc_full, axis=1)
    res_u_mpc = np.concatenate(res_u_mpc, axis=1)

    return res_x_mpc, res_x_mpc_full, res_u_mpc


if __name__ == "__main__":
    params_dict = {
        "N_time": 30,
        "dt": 0.04,
        "max_solver_iter": 35,
        "MPC_K": 2,
        "MPC_N": 40,
        # x
        "MPC_Q_0": 1e-3,
        # xdot
        "MPC_Q_1": 1e-8,
        # theta
        "MPC_Q_2": 1e-1,
        # thetadot
        "MPC_Q_3": 1e-8,
        # importance
        "MPC_R": 1e-4,
        "MPC_x_bound": 50.0,
        "MPC_xdot_bound": 20.0,
        "MPC_theta_bound": 1.6 * 3.1415,
        "MPC_thetadot_bound": 100.0,
        "MPC_u_bound": 30,
    }
    x_0 = np.array([0.5, 0, -1.56, 0, 0.36, 0.26]).reshape([-1, 1])  # true states
    res_x_mpc, res_x_mpc_full, res_u_mpc = simulate(params_dict, x_0)
    plot_trajectories(res_x_mpc, res_u_mpc)
    animate_system(res_x_mpc, init=x_0[:4], dt=params_dict["dt"])
