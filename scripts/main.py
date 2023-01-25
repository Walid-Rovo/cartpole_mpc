import numpy as np

from simulate import simulate
from plots import visualize_ekf, plot_trajectories


if __name__ == "__main__":
    params_dict = {
        "N_time": 20,
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
