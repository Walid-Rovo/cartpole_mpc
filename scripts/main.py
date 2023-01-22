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


def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, raise_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def raise_timeout(signum, frame):
    raise TimeoutError("Function timed out")


@timeout(25)
def simulate(params_dict, x_0=np.array([0.5, 0, 0.0, 0, 0.36, 0.26]).reshape([-1, 1])):
    # Define the initial state
    # x_0 = np.array([0.5, 0, 3.1, 0, 0.1, 0.1]).reshape([-1, 1])  # required

    # configure observer
    # configure observer
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
        R=params_dict["MPC_R"],
        dt=params_dict["dt"],
        u_bound=params_dict["MPC_u_bound"],
    )
    controller = solver.generate_solver()

    # configure simulator
    pendulum = cartpole_sim.PendulumOnCart(
        initial_states=x_0[:4], dt=params_dict["dt"], render=False
    )
    # loop Variables
    # Initialize result lists for states and inputs
    res_x_mpc = [x_0[:4]]
    res_x_mpc_full = [x_0]
    res_x_hat = [x_0]
    res_u_mpc = []

    # Set number of iterations
    N_time = params_dict["N_time"]
    N_sim = int(N_time / params_dict["dt"])
    u_k = 0
    # simulation loop
    for _ in range(N_sim):
        x_hat_full = observer.discrete_EKF_filter(y=np.array([x_0[0], x_0[2]]), u=u_k)
        x_hat = x_hat_full[:4]

        # solve optimization problem
        solver.update_state(x_hat)
        mpc_res = controller(p=x_hat, lbg=0, ubg=0, lbx=solver.lb_opt_x, ubx=solver.ub_opt_x)

        # Extract the control input
        opt_x_k = solver.opt_x(mpc_res["x"])
        u_k = opt_x_k["u", 0]

        # simulate the system
        x_next = pendulum.step(action=u_k)

        # Update the initial state
        x_0 = x_next

        # Store the results
        res_x_mpc.append(x_next)
        res_x_mpc_full.append(np.concatenate((x_next, x_0[:-2])))
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
    #     "N_time": 10,
    #     "dt": 0.02,
    #     "EKF_P0_0": 1e-06,
    #     "EKF_P0_1": 1e-06,
    #     "EKF_P0_2": 0.0001,
    #     "EKF_P0_3": 0.01,
    #     "EKF_P0_4": 1e-06,
    #     "EKF_P0_5": 0.0001,
    #     "EKF_Q_0": 0.01,
    #     "EKF_Q_1": 10.0,
    #     "EKF_Q_2": 10.0,
    #     "EKF_Q_3": 10.0,
    #     "EKF_Q_4": 0.0001,
    #     "EKF_Q_5": 0.01,
    #     "EKF_R_0": 1e-08,
    #     "EKF_R_1": 0.0001,
    #     "MPC_K": 3,
    #     "MPC_N": 45,
    #     "MPC_Q_0": 0.001,
    #     "MPC_Q_1": 0.1,
    #     "MPC_Q_2": 0.001,
    #     "MPC_Q_3": 1e-05,
    #     "MPC_R": 0.01,
    #     "MPC_u_bound": 5,
    # }
    # params_dict = {
    #     "N_time": 5,
    #     "dt": 0.02,
    #     "EKF_P0_0": 0.01,
    #     "EKF_P0_1": 0.0001,
    #     "EKF_P0_2": 1e-06,
    #     "EKF_P0_3": 1e-06,
    #     "EKF_P0_4": 0.0001,
    #     "EKF_P0_5": 1e-06,
    #     "EKF_Q_0": 0.0001,
    #     "EKF_Q_1": 0.0001,
    #     "EKF_Q_2": 10.0,
    #     "EKF_Q_3": 0.0001,
    #     "EKF_Q_4": 0.0001,
    #     "EKF_Q_5": 10.0,
    #     "EKF_R_0": 0.0001,
    #     "EKF_R_1": 1e-06,
    #     "MPC_K": 2,
    #     "MPC_N": 30,
    #     "MPC_Q_0": 0.001,
    #     "MPC_Q_1": 0.1,
    #     "MPC_Q_2": 0.1,
    #     "MPC_Q_3": 0.1,
    #     "MPC_R": 0.0001,
    #     "MPC_u_bound": 5,
    # }
    params_dict = {
        "N_time": 15,
        "dt": 0.02,
        "EKF_P0_0": 0.0001,
        "EKF_P0_1": 0.0001,
        "EKF_P0_2": 0.01,
        "EKF_P0_3": 1e-06,
        "EKF_P0_4": 0.01,
        "EKF_P0_5": 0.01,
        "EKF_Q_0": 0.01,
        "EKF_Q_1": 0.0001,
        "EKF_Q_2": 0.0001,
        "EKF_Q_3": 10.0,
        "EKF_Q_4": 10.0,
        "EKF_Q_5": 10.0,
        "EKF_R_0": 0.0001,
        "EKF_R_1": 1e-08,
        "MPC_K": 2,
        "MPC_N": 45,
        "MPC_Q_0": 0.1,
        "MPC_Q_1": 1e-05,
        "MPC_Q_2": 1e-05,
        "MPC_Q_3": 1e-05,
        "MPC_R": 1e-06,
        "MPC_u_bound": 50,
    }
    # params_dict = {
    #     "N_time": 5,
    #     "dt": 0.02,
    #     "EKF_P0_0": 0.01,
    #     "EKF_P0_1": 0.01,
    #     "EKF_P0_2": 0.0001,
    #     "EKF_P0_3": 0.0001,
    #     "EKF_P0_4": 0.01,
    #     "EKF_P0_5": 0.0001,
    #     "EKF_Q_0": 0.0001,
    #     "EKF_Q_1": 0.0001,
    #     "EKF_Q_2": 10.0,
    #     "EKF_Q_3": 0.01,
    #     "EKF_Q_4": 0.01,
    #     "EKF_Q_5": 0.01,
    #     "EKF_R_0": 1e-08,
    #     "EKF_R_1": 1e-08,
    #     "MPC_K": 2,
    #     "MPC_N": 45,
    #     "MPC_Q_0": 0.1,
    #     "MPC_Q_1": 1e-05,
    #     "MPC_Q_2": 1e-05,
    #     "MPC_Q_3": 0.1,
    #     "MPC_R": 0.01,
    #     "MPC_u_bound": 25,
    # }
    # params_dict = {
    #     "N_time": 5,
    #     "dt": 0.02,
    #     "EKF_P0_0": 1e-06,
    #     "EKF_P0_1": 1e-06,
    #     "EKF_P0_2": 0.0001,
    #     "EKF_P0_3": 1e-06,
    #     "EKF_P0_4": 0.0001,
    #     "EKF_P0_5": 0.01,
    #     "EKF_Q_0": 0.0001,
    #     "EKF_Q_1": 0.0001,
    #     "EKF_Q_2": 10.0,
    #     "EKF_Q_3": 0.0001,
    #     "EKF_Q_4": 0.0001,
    #     "EKF_Q_5": 0.01,
    #     "EKF_R_0": 1e-06,
    #     "EKF_R_1": 1e-08,
    #     "MPC_K": 2,
    #     "MPC_N": 30,
    #     "MPC_Q_0": 1e-05,
    #     "MPC_Q_1": 0.1,
    #     "MPC_Q_2": 0.1,
    #     "MPC_Q_3": 0.1,
    #     "MPC_R": 0.0001,
    #     "MPC_u_bound": 5,
    # }
    x_0 = np.array([0.5, 0, 0.0, 0, 0.36, 0.26]).reshape([-1, 1])  # true states
    res_x_mpc, res_x_mpc_full, res_x_hat, res_u_mpc = simulate(params_dict, x_0)
    plot_trajectories(res_x_mpc, res_u_mpc)
    visualize_ekf(res_x_mpc_full, res_x_hat)
    animate_system(res_x_mpc, init=x_0[:4], dt=params_dict["dt"])
