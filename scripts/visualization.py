import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

import cartpole_sim

# configure plotter
mpl.rcParams["font.size"] = 16


# function for visualizing control and state trajectories
def plot_trajectories(traj_x, traj_u):
    # plot states and input
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    # plot the states
    ax[0].plot(traj_x.T, label=(r"$x$", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"))
    ax[0].legend(loc="upper right", fontsize="xx-small")
    ax[1].plot(traj_u.T)
    # Set labels
    ax[0].set_ylabel("states")
    ax[1].set_ylabel("inputs")
    ax[1].set_xlabel("time")
    plt.show()


# function to display animated system
def animate_system(traj_x, init, dt=0.02):
    anime = cartpole_sim.PendulumOnCart(initial_states=init, dt=dt, render=True)

    for i in range(traj_x.shape[1]):
        anime.set_states(traj_x[:, i])
        anime.render()
        time.sleep(dt)
