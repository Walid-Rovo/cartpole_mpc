import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import mpc
import cartpole_sim


def simulate():
    # configure plotter
    mpl.rcParams['font.size'] = 16

    # system constants
    # Dimensions of x and u:
    nx = 4
    nu = 1
    DT = 0.02

    # configure controller
    solver = mpc.MPC(dt=DT)
    controller = solver.generate_solver()

    # configure simulator
    pendulum = cartpole_sim.PendulumOnCart(dt=DT, render=True)
    pendulum.reset()

    # loop Variables
    # Define the initial state
    # x_0 = np.array([0, 0.2, np.pi / 2, 0]).reshape(nx, 1)
    x_0 = np.array([0, 0.2, np.pi / 2, 0]).reshape(nx, 1)

    # Initialize result lists for states and inputs
    res_x_mpc = [x_0]
    res_u_mpc = []

    # Set number of iterations
    N_time = 10
    N_sim = int(N_time / DT)

    # simulation loop
    for i in range(N_sim):
        # solve optimization problem
        mpc_res = controller(p=x_0, lbg=0, ubg=0, lbx=solver.lb_opt_x, ubx=solver.ub_opt_x)

        # Extract the control input
        opt_x_k = solver.opt_x(mpc_res['x'])
        u_k = opt_x_k['u', 0]

        # simulate the system
        x_next = pendulum.step(action=u_k)

        # 04 - Your code here!
        # Update the initial state
        x_0 = x_next
        # 04

        # 05 - Your code here!
        # Store the results
        res_x_mpc.append(x_next)
        res_u_mpc.append(u_k)
        # 05

    # Make an array from the list of arrays:
    res_x_mpc = np.concatenate(res_x_mpc, axis=1)
    res_u_mpc = np.concatenate(res_u_mpc, axis=1)

    # plot states and input
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    # plot the states
    ax[0].plot(res_x_mpc.T, label=(r"$x$", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"))
    ax[0].legend(loc="upper right", fontsize="xx-small")
    ax[1].plot(res_u_mpc.T)
    # Set labels
    ax[0].set_ylabel('states')
    ax[1].set_ylabel('inputs')
    ax[1].set_xlabel('time')

    fig.show()


if __name__ == "__main__":
    simulate()
