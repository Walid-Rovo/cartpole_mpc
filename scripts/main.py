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

np.random.seed(99)


def simulate(
    params_dict,
    x_0_plant=np.array([0.5, 0, 3.1, 0, 0.36, 0.23]).reshape([-1, 1]),
    x_0_observer=np.array([-0.5, 0, 2.8, 0, 0.1, 0.1]).reshape([-1, 1]),
    render=False,
):
    # Configure observer
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

    # Configure simulator
    pendulum = cartpole_sim.PendulumOnCart(
        initial_states=x_0_plant[:4], dt=params_dict["dt"], render=render
    )

    # Initialize result lists for states and inputs
    res_x_mpc = [x_0_observer[:4]]
    res_x_mpc_full = [x_0_plant]
    res_x_hat = [x_0_observer]
    res_u_mpc = []
    res_std_ekf = [np.sqrt(np.diag(observer.P).reshape(-1, 1))]
    res_ref_mpc = [np.array([[0]])]

    # Iteration setup
    N_time = params_dict["N_time"]
    N_sim = int(N_time / params_dict["dt"])
    excite_u = 50.0
    u_k = np.array([[excite_u]])
    x_next = x_0_observer[:4]

    for k in range(N_sim):
        # Measurement noise
        wy_0 = np.random.normal(0, np.sqrt(params_dict["wy_0"]))
        wy_2 = np.random.normal(0, np.sqrt(params_dict["wy_2"]))
        # wy_0 = 0
        # wy_2 = 0

        # EKF step with noise
        x_hat_full = observer.discrete_EKF_filter(
            y=np.array([x_0_observer[0] + wy_0, x_0_observer[2] + wy_2]), u=u_k
        )

        # We only want 4 states for MPC
        x_hat = x_hat_full[:4]

        # Only turn on the MPC after two seconds, to allow the EKF to reach good estimates
        # EXCITE_SECONDS = 0
        # if k % int(1 / params_dict["dt"]) == 0 and k <= int(EXCITE_SECONDS / params_dict["dt"]):
        #     print("Exciting")
        #     excite_u *= -1.5
        #     u_k = np.array([[excite_u]])
        # elif k > int(EXCITE_SECONDS / params_dict["dt"]):
        solver.update_state(x_hat)
        mpc_res = solver.controller(
            p=x_next, lbg=0, ubg=0, lbx=solver.lb_opt_x, ubx=solver.ub_opt_x
        )
        # Extract the control input
        opt_x_k = solver.opt_x(mpc_res["x"])
        u_k = opt_x_k["u", 0].full()
        # else:
        #     u_k = np.zeros_like(u_k)

        if k == int(7 / params_dict["dt"]):
            # Change x setpoint
            solver.setpoint[0] = 4
            solver.generate()
        elif k > int(7 / params_dict["dt"]) and k % int(10 / params_dict["dt"]) == 0:
            # Flip x setpoint
            solver.setpoint[0] *= -1
            solver.generate()

        # Simulate the system
        x_next = pendulum.step(action=u_k)

        # Update the initial state
        x_0_observer[:4] = x_next

        # Store the results
        res_x_mpc.append(x_next)
        res_x_mpc_full.append(np.concatenate((x_next, x_0_plant[-2:])))
        res_x_hat.append(x_hat_full)
        res_u_mpc.append(u_k)
        res_std_ekf.append(np.sqrt(np.diag(observer.P).reshape(-1, 1)))
        res_ref_mpc.append(np.array([solver.setpoint[0]]))

    # Make a numpy array from the list of arrays
    res_x_mpc = np.concatenate(res_x_mpc, axis=1)
    res_x_mpc_full = np.concatenate(res_x_mpc_full, axis=1)
    res_x_hat = np.concatenate(res_x_hat, axis=1)
    res_u_mpc = np.concatenate(res_u_mpc, axis=1)
    res_std_ekf = np.concatenate(res_std_ekf, axis=1)
    res_ref_mpc = np.concatenate(res_ref_mpc, axis=1)
    res_x_mpc = np.concatenate((res_ref_mpc, res_x_mpc), axis=0)

    print(f"Last P for observer: {observer.P}")

    return res_x_mpc, res_x_mpc_full, res_x_hat, res_u_mpc, res_std_ekf


if __name__ == "__main__":
    params_dict = {
        "N_time": 40,
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
        # x
        "wy_0": 1e-6,
        # theta
        "wy_2": 1e-6,
        "max_solver_iter": 40,
        "MPC_K": 2,
        "MPC_N": 40,
        # x
        "MPC_Q_0": 1e-2,  #
        # xdot
        "MPC_Q_1": 1e-6,
        # theta
        "MPC_Q_2": 1e-5,  #
        # thetadot
        "MPC_Q_3": 1e-9,
        # importance
        "MPC_R": 1e-4,  #
        "MPC_x_bound": 10.0,
        "MPC_xdot_bound": 20.0,
        "MPC_theta_bound": 1.0 * 3.7698,  #
        "MPC_thetadot_bound": 0.7,  #
        "MPC_u_bound": 35,
    }

    x_0_plant = np.array([0.5, 0, 3.1, 0, 0.36, 0.23]).reshape([-1, 1])  # PDF plant states
    x_0_observer = np.array([-0.5, 0, 2.8, 0, 0.1, 0.1]).reshape([-1, 1])  # PDF observer states

    res_x_mpc, res_x_mpc_full, res_x_hat, res_u_mpc, res_std_ekf = simulate(
        params_dict, x_0_plant, x_0_observer, render=True
    )
    plot_trajectories(res_x_mpc, res_u_mpc, params_dict["dt"])
    visualize_ekf(res_x_mpc_full, res_x_hat, res_std_ekf, params_dict["dt"])
