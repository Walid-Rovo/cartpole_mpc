from functools import wraps
import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from casadi import *

from mpc import MPC
import cartpole_sim
from observer import EKF
from visualization import plot_trajectories, visualize_ekf, animate_system


def simulate_calibrate(params_dict, x_0=np.array([0.0, 0, 0, 0, 0.36, 0.26]).reshape([-1, 1])):
    # Configure observer
    observer = EKF(
        x0=x_0,
        P0=np.diag(
            [
                params_dict["EKF_P0_0"],
                params_dict["EKF_P0_1"],
                params_dict["EKF_P0_2"],
                params_dict["EKF_P0_3"],
                params_dict["EKF_P0_4"],
                params_dict["EKF_P0_5"],
            ]
        ),
        Q=np.diag(
            [
                params_dict["EKF_Q_0"],
                params_dict["EKF_Q_1"],
                params_dict["EKF_Q_2"],
                params_dict["EKF_Q_3"],
                params_dict["EKF_Q_4"],
                params_dict["EKF_Q_5"],
            ]
        ),
        R=np.diag(
            [
                params_dict["EKF_R_0"],
                params_dict["EKF_R_1"],
            ]
        ),
        dt=params_dict["dt"],
    )

    # configure simulator
    pendulum = cartpole_sim.PendulumOnCart(
        initial_states=x_0[:4], dt=params_dict["dt"], render=False
    )
    # loop Variables
    # Initialize result lists for states and inputs
    res_x_mpc = [x_0[:4]]
    res_x_mpc_full = [x_0]
    res_x_hat = [x_0]
    res_u_mpc = [np.array([1.])]
    res_u_mpc = [np.array([0])]

    # Set number of iterations
    N_time = params_dict["N_time"]
    N_sim = int(N_time / params_dict["dt"])
    u_k = res_u_mpc[0]
    # simulation loop
    x_next = x_0[:4]
    for k in range(N_sim):
        x_hat_full = observer.discrete_EKF_filter(y=np.array([x_0[0], x_0[2]]), u=u_k)

        # # Generate excitation force
        # if not k % 40:
        #     u_k = np.random.normal(0, 1e1) + -abs(np.random.normal(0, 1e1)) * x_0[1]
        # else:
        #     u_k = np.zeros_like(u_k)

        # simulate the system
        x_next = pendulum.step(action=u_k)

        # Update the initial state
        x_0[:4] = x_next

        # Store the results
        res_x_mpc.append(x_next)
        res_x_mpc_full.append(np.concatenate([x_next, x_0[-2:]]))
        res_x_hat.append(x_hat_full)
        res_u_mpc.append(u_k)

    # Make an array from the list of arrays:
    res_x_mpc = np.concatenate(res_x_mpc, axis=1)
    res_x_mpc_full = np.concatenate(res_x_mpc_full, axis=1)
    res_x_hat = np.concatenate(res_x_hat, axis=1)
    res_u_mpc = np.concatenate(res_u_mpc, axis=0)

    return res_x_mpc, res_x_mpc_full, res_x_hat, res_u_mpc,\
           observer.P, observer.xhat, x_next

def simulate(params_dict, x_0_plant, x_0_observer):
    # Define the initial state
    # x_0 = np.array([0.5, 0, 3.1, 0, 0.1, 0.1]).reshape([-1, 1])  # required

    # configure observer
    # configure observer
    observer = EKF(
        x0=x_0_observer,
        P0=np.diag(
            [
                params_dict["EKF_P0_0"],
                params_dict["EKF_P0_1"],
                params_dict["EKF_P0_2"],
                params_dict["EKF_P0_3"],
                params_dict["EKF_P0_4"],
                params_dict["EKF_P0_5"],
            ]
        ),
        Q=np.diag(
            [
                params_dict["EKF_Q_0"],
                params_dict["EKF_Q_1"],
                params_dict["EKF_Q_2"],
                params_dict["EKF_Q_3"],
                params_dict["EKF_Q_4"],
                params_dict["EKF_Q_5"],
            ]
        ),
        R=np.diag(
            [
                params_dict["EKF_R_0"],
                params_dict["EKF_R_1"],
            ]
        ),
        dt=params_dict["dt"],
    )

    # configure controller
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
        Qf=np.diag(
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
    )
    solver.generate()

    # configure simulator
    pendulum = cartpole_sim.PendulumOnCart(
        initial_states=x_0_plant[:4], dt=params_dict["dt"], render=True
    )
    # loop Variables
    # Initialize result lists for states and inputs
    res_x_mpc = [x_0_observer[:4]]
    res_x_mpc_full = [x_0_observer]
    res_x_hat = [x_0_observer]
    res_u_mpc = []

    # Set number of iterations
    N_time = params_dict["N_time"]
    N_sim = int(N_time / params_dict["dt"])
    u_k = np.array([[0.0]])
    # simulation loop
    x_next = x_0_observer[:4]
    for k in range(N_sim):
        x_hat_full = observer.discrete_EKF_filter(y=np.array([x_0_observer[0], x_0_observer[2]]), u=u_k)
        x_hat = x_hat_full[:4]

        # solve optimization problem
        if k > 1:
            solver.update_state(x_hat)
            mpc_res = solver.controller(p=x_next, lbg=0, ubg=0, lbx=solver.lb_opt_x, ubx=solver.ub_opt_x)
            # extract the control input
            opt_x_k = solver.opt_x(mpc_res["x"])
            u_k = opt_x_k["u", 0].full()
        elif k % 25 and k < 1 :
            u_k = (np.random.normal(0, 1e1) + -abs(np.random.normal(0, 1e1)) * x_next[1]).reshape(-1, 1)
        else:
            u_k = np.zeros_like(u_k)

        # simulate the system
        x_next = pendulum.step(action=u_k)

        # Update the initial state
        x_0_observer[:4] = x_next

        # Store the results
        res_x_mpc.append(x_next)
        res_x_mpc_full.append(np.concatenate((x_next, x_0_observer[-2:])))
        res_x_hat.append(x_hat_full)
        res_u_mpc.append(u_k)

    # Make an array from the list of arrays:
    res_x_mpc = np.concatenate(res_x_mpc, axis=1)
    res_x_mpc_full = np.concatenate(res_x_mpc_full, axis=1)
    res_x_hat = np.concatenate(res_x_hat, axis=1)
    res_u_mpc = np.concatenate(res_u_mpc, axis=1)

    return res_x_mpc, res_x_mpc_full, res_x_hat, res_u_mpc


if __name__ == "__main__":
    # params_dict = {
    #     "N_time": 15,
    #     "dt": 0.04,
    #     "EKF_P0_0": 1e-6,
    #     "EKF_P0_1": 1e-6,
    #     "EKF_P0_2": 1e-6,
    #     "EKF_P0_3": 1e-6,
    #     "EKF_P0_4": 1e-6,
    #     "EKF_P0_5": 1e-6,
    #     "EKF_Q_0": 1e-6,
    #     "EKF_Q_1": 1e-6,
    #     "EKF_Q_2": 1e-6,
    #     "EKF_Q_3": 1e-3,
    #     "EKF_Q_4": 1e-3,
    #     "EKF_Q_5": 1e-3,
    #     "EKF_R_0": 1e-6,
    #     "EKF_R_1": 1e-6,

    #     "max_solver_iter": 40,
    #     "MPC_K": 2,
    #     "MPC_N": 40,
    #     # x
    #     "MPC_Q_0": 1e-3,
    #     # xdot
    #     "MPC_Q_1": 1e-8,
    #     # theta
    #     "MPC_Q_2": 1e-1,
    #     # thetadot
    #     "MPC_Q_3": 1e-9,
    #     # importance
    #     "MPC_R": 1e-4,
    #     "MPC_x_bound": 50.0,
    #     "MPC_xdot_bound": 10.0,
    #     "MPC_theta_bound": 1 * 3.1415,
    #     "MPC_thetadot_bound": 0.75,
    #     "MPC_u_bound": 50,
    # }
    params_dict = {
        "N_time": 30,
        "dt": 0.04,
        "EKF_P0_0": 1e-6,
        "EKF_P0_1": 1e-6,
        "EKF_P0_2": 1e-6,
        "EKF_P0_3": 1e-6,
        "EKF_P0_4": 1e-6,
        "EKF_P0_5": 1e-6,
        "EKF_Q_0": 1e-6,
        "EKF_Q_1": 1e-6,
        "EKF_Q_2": 1e-6,
        "EKF_Q_3": 1e-3,
        "EKF_Q_4": 1e-3,
        "EKF_Q_5": 1e-3,
        "EKF_R_0": 1e-6,
        "EKF_R_1": 1e-6,

        "max_solver_iter": 40,
        "MPC_K": 2,
        "MPC_N": 40,
        # x
        "MPC_Q_0": 1e-3,
        # xdot
        "MPC_Q_1": 1e-8,
        # theta
        "MPC_Q_2": 1e-1,
        # thetadot
        "MPC_Q_3": 1e-9,
        # importance
        "MPC_R": 1e-4,
        "MPC_x_bound": 50.0,
        "MPC_xdot_bound": 20.0,
        "MPC_theta_bound": 1.2 * 3.1415,
        "MPC_thetadot_bound": 1.2,
        "MPC_u_bound": 30,
    }
    # params_dict = {
    #     "N_time": 10,
    #     "dt": 0.02,
    #     "EKF_P0_0": 1e-5,
    #     "EKF_P0_1": 1e-6,
    #     "EKF_P0_2": 1e-1,
    #     "EKF_P0_3": 1e-3,
    #     "EKF_P0_4": 1e-1,
    #     "EKF_P0_5": 1e-1,
    #     "EKF_Q_0": 1e-5,
    #     "EKF_Q_1": 1e-6,
    #     "EKF_Q_2": 1e-1,
    #     "EKF_Q_3": 1e-3,
    #     "EKF_Q_4": 1e-1,
    #     "EKF_Q_5": 1e-1,
    #     "EKF_R_0": 1e-7,
    #     "EKF_R_1": 1e-7,

    #     "max_solver_iter": 50,
    #     "MPC_K": 2,
    #     "MPC_N": 50,
    #     # x
    #     "MPC_Q_0": 1e-3,
    #     # xdot
    #     "MPC_Q_1": 1e-8,
    #     # theta
    #     "MPC_Q_2": 1e-3,
    #     # thetadot
    #     "MPC_Q_3": 1e-9,
    #     # importance
    #     "MPC_R": 5e-2,
    #     "MPC_x_bound": 4.0,
    #     "MPC_xdot_bound": 3.0,
    #     "MPC_theta_bound": 1.3 * 3.1415,
    #     "MPC_thetadot_bound": 0.40,
    #     "MPC_u_bound": 40,
    # }
    # x_0 = np.array([0.5, 0, -3, 0, 0.36, 0.26]).reshape([-1, 1])  # true states
    x_0 = np.array([0.5, 0, 1, 0, 0.36, 0.23]).reshape([-1, 1])  # PDF states
    x_0_plant = x_0
    x_0_observer = x_0
    # x_0_plant = np.array([0.5, 0, 3.1, 0, 0.36, 0.23]).reshape([-1, 1])  # PDF states
    # x_0_observer = np.array([-0.5, 0, 2.8, 0, 0.1, 0.1]).reshape([-1, 1])  # PDF states
    res_x_mpc, res_x_mpc_full, res_x_hat, res_u_mpc = simulate(params_dict, x_0_plant, x_0_observer)
    plot_trajectories(res_x_mpc, res_u_mpc)
    visualize_ekf(res_x_mpc_full, res_x_hat)
    # animate_system(res_x_mpc, init=x_0[:4], dt=params_dict["dt"])
