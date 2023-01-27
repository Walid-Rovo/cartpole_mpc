import numpy as np

from simulate import simulate
from plots import visualize_ekf, plot_trajectories


if __name__ == "__main__":
    params_dict = {
        "seed": 3875886,
        "N_time": 7,
        "dt": 0.04,
        "EKF_P0_0": 1e-06,
        "EKF_P0_1": 1e-06,
        "EKF_P0_2": 1e-06,
        "EKF_P0_3": 1e-06,
        "EKF_P0_4": 1e-06,
        "EKF_P0_5": 1e-06,
        "EKF_Q_0": 1e-06,
        "EKF_Q_1": 1e-06,
        "EKF_Q_2": 1e-06,
        "EKF_Q_3": 1e-06,
        "EKF_Q_4": 0.001,
        "EKF_Q_5": 0.001,
        "EKF_R_0": 1e-06,
        "EKF_R_1": 1e-06,
        "wy_0": 1e-06,
        "wy_2": 1e-06,
        "max_solver_iter": 40,
        "MPC_K": 2,
        "MPC_N": 60,
        "MPC_Q_0": 0.1,
        "MPC_Q_1": 1e-06,
        "MPC_Q_2": 0.001,
        "MPC_Q_3": 0.1,
        "MPC_R": 0.001,
        "MPC_x_bound": 10.0,
        "MPC_xdot_bound": 30.0,
        "MPC_theta_bound": 3.1415,
        "MPC_thetadot_bound": 0.3,
        "MPC_u_bound": 25,
    }
    x_0_plant = np.array([0.5, 0, 3.1, 0, 0.36, 0.23]).reshape([-1, 1])  # PDF plant states
    x_0_observer = np.array([-0.5, 0, 2.8, 0, 0.1, 0.1]).reshape([-1, 1])  # PDF observer states

    res_x_mpc, res_x_mpc_full, res_x_hat, res_u_mpc, res_std_ekf, ekf_P = simulate(
        params_dict, x_0_plant, x_0_observer, render=True, control=False
    )
    print(f'MSE of true state vs estimated state: \n{np.square(res_x_mpc_full[:, :] - res_x_hat[:, :]).mean(axis=1)}')
    np.set_printoptions(precision=3)
    print(f"Observer P diagonal: \n{np.diag(ekf_P.full())}")
    plot_trajectories(res_x_mpc, res_u_mpc, params_dict["dt"])
    visualize_ekf(res_x_mpc_full, res_x_hat, res_std_ekf, params_dict["dt"])
