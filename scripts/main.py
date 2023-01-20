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

    # configure controller
    solver = mpc.MPC()
    controller = solver.generate_solver()

    # configure simulator
    pendulum = cartpole_sim.PendulumOnCart(render=True)

    # loop Variables
    # Define the initial state
    x_0 = np.array([0, 0.2, np.pi / 2, 0]).reshape(nx, 1)

    # Initialize result lists for states and inputs
    res_x_mpc = [x_0]
    res_u_mpc = []

    # Set number of iterations
    N_sim = 1000

    # simulation loop
    for i in range(N_sim):

        # solve optimization problem
        mpc_res = controller(p=x_0, lbg=0, ubg=0, lbx=solver.lb_opt_x, ubx=solver.ub_opt_x)
        # optionally: Warmstart the optimizer by passing the previous solution as an initial guess!
        if i > 0:
            mpc_res = controller(p=x_0, x0=opt_x_k, lbg=solver.lb_g, ubg=solver.ub_g, lbx=solver.lb_opt_x, ubx=solver.ub_opt_x)
        # Extract the control input
        opt_x_k = solver.opt_x(mpc_res['x'])
        u_k = opt_x_k['u', 0]

        # simulate the system
        res_integrator = pendulum.step(action=u_k)
        x_next = res_integrator['xf']

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

if __name__ == "__main__":
    simulate()
