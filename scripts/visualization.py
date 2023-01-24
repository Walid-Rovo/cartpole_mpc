import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import time

import cartpole_sim

# configure plotter
mpl.rcParams["font.size"] = 16


# function for visualizing control and state trajectories
def plot_trajectories(traj_x, traj_u, dt):
    # plot states and input
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    # plot the states
    time = dt * np.arange(0, traj_x.shape[-1], 1).reshape(-1, 1)
    ax[0].plot(time, traj_x.T, label=(r"$ref \ (m)$", r"$x \ (m)$", r"$\dot{x} \ (\frac{m}{s})$", r"$\theta \ (rad)$", r"$\dot{\theta} \ (\frac{rad}{s})$"))
    ax[0].legend(loc="lower right", fontsize="xx-small")
    ax[1].plot(time[:-1], traj_u.T)
    # Set labels
    ax[0].set_ylabel("states")
    ax[1].set_ylabel("inputs")
    ax[1].set_xlabel("time (s)")
    ax[0].legend(loc=1, prop={"size": 14})
    plt.show()


# function to display animated system
def animate_system(traj_x, init, dt=0.02):
    anime = cartpole_sim.PendulumOnCart(initial_states=init, dt=dt, render=True)

    for i in range(traj_x.shape[1]):
        anime.set_states(traj_x[:, i])
        anime.render()
        time.sleep(dt)


def visualize_ekf(x_data, x_hat_data, res_std_ekf, dt):
    nx = x_data.shape[0]
    fig, ax = plt.subplots(nx)
    fig.suptitle("EKF Observer")

    N_STD = 3

    time = dt * np.arange(0, res_std_ekf[0].shape[-1], 1).T

    for i in range(nx):
        ax[i].plot(time, x_data[i, :], label="$x$")
        ax[i].plot(time, x_hat_data[i, :], "r--", label="$\hat{x}$")
        ax[i].set_ylabel("x{}".format(i))
        ax[i].fill_between(
            time,
            x_hat_data[i, :] + (N_STD * np.array(res_std_ekf[i, :])),
            x_hat_data[i, :] - (N_STD * np.array(res_std_ekf[i, :])),
            color='grey',
            alpha=0.4,
            label="$\pm 3 \sigma$",
        )
    ax[0].plot(label="$\Delta{t}$ = ")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop=fm.FontProperties(size=14))
    ax[-1].set_xlabel("$time (s)$")
    plt.show()
