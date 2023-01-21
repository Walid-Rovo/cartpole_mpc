import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from casadi import *

from mpc import MPC
import cartpole_sim
from observer import EKF
from visualization import *


def simulate():
    # system constants
    # Dimensions of x and u:
    nx = 4
    nu = 1
    DT = 0.05

    # Define the initial state
    # x_0 = np.array([0, 0, 0, 0]).reshape(nx, 1)
    # x_0 = np.array([0, 0.2, np.pi / 2, 0, 0.36, 0.23]).reshape([-1, 1])
    x_0 = np.array([0.5, 0, 3.1, 0, 0.1, 0.1]).reshape([-1, 1])

    # configure observer
    observer = EKF(x0=x_0, dt=DT)

    # configure controller
    solver = MPC(dt=DT)
    controller = solver.generate_solver()

    # configure simulator
    pendulum = cartpole_sim.PendulumOnCart(initial_states=x_0[:4], dt=DT, render=False)

    # loop Variables
    # Initialize result lists for states and inputs
    res_x_mpc = [x_0[:4]]
    res_x_hat = [x_0[:4]]
    res_u_mpc = []

    # Set number of iterations
    N_time = 10
    N_sim = int(N_time / DT)

    # simulation loop
    for i in range(N_sim):
        x_hat = observer.discrete_EKF_filter(y=np.array([x_0[0], x_0[2]]), u=0)[:4]

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
        res_x_hat.append(x_hat)
        res_u_mpc.append(u_k)

    # Make an array from the list of arrays:
    res_x_mpc = np.concatenate(res_x_mpc, axis=1)
    res_x_hat = np.concatenate(res_x_hat, axis=1)
    res_u_mpc = np.concatenate(res_u_mpc, axis=1)

    plot_trajectories(res_x_mpc, res_u_mpc)
    animate_system(res_x_mpc, init=x_0, dt=DT)


if __name__ == "__main__":
    simulate()
