from concurrent.futures import ThreadPoolExecutor
import datetime
from functools import wraps
import json
from math import inf
import os
import signal
import sys
import numpy as np

from mpc import MPC
import cartpole_sim
from observer import EKF

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

@timeout(20)
def simulate(params_dict):
    # Define the initial state
    x_0 = np.array([0.5, 0, 3.1, 0, 0.1, 0.1]).reshape([-1, 1])  # required
    # x_0 = np.array([0.5, 0, 3.1, 0, 0.36, 0.23]).reshape([-1, 1])  # true states

    # configure observer
    observer = EKF(
        x0=x_0,
        P0=np.diag([
            params_dict['EKF_P0_0'],
            params_dict['EKF_P0_1'],
            params_dict['EKF_P0_2'],
            params_dict['EKF_P0_3'],
            params_dict['EKF_P0_4'],
            params_dict['EKF_P0_5'],
        ]),
        Q=np.diag([
            params_dict['EKF_Q_0'],
            params_dict['EKF_Q_1'],
            params_dict['EKF_Q_2'],
            params_dict['EKF_Q_3'],
            params_dict['EKF_Q_4'],
            params_dict['EKF_Q_5'],
        ]),
        R=np.diag([
            params_dict['EKF_R_0'],
            params_dict['EKF_R_1'],
        ]),
        dt=params_dict['dt'],
    )

    # configure controller
    solver = MPC(
        K=params_dict['MPC_K'],
        N=params_dict['MPC_N'],
        Q=np.diag([
            params_dict['MPC_Q_0'],
            params_dict['MPC_Q_1'],
            params_dict['MPC_Q_2'],
            params_dict['MPC_Q_3'],
        ]),
        R=params_dict['MPC_R'],
        dt=params_dict['dt'],
        u_bound=params_dict['MPC_u_bound'],
    )
    controller = solver.generate_solver()

    # configure simulator
    pendulum = cartpole_sim.PendulumOnCart(
        initial_states=x_0[:4],
        dt=params_dict['dt'],
        render=False
    )

    # Set number of iterations
    N_time = params_dict['N_time']
    N_sim = int(N_time / params_dict['dt'])
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

    return x_0

def iterate_simulate(thread_number, list_dicts, filename):
    log_filename = "./search_results/result-" + filename + ".log"
    lowest_norm = inf
    for idx, params_dict in enumerate(list_dicts):
        try:
            x = simulate(params_dict)
        except (TimeoutError, RuntimeError) as e:
            print(f"[{thread_number}] Iteration {idx} errored: {str(e)}\n\n")
            continue
        norm = np.linalg.norm(x[:4])
        if norm < lowest_norm:
            lowest_norm = norm
            log_message = (
                f"[{thread_number}] Iteration {idx}/{len(list_dicts)}\n"
                f"[{thread_number}] Reached a new lowest norm: {lowest_norm}\n"
                f"[{thread_number}] Final state: \n{x}\n"
                f"[{thread_number}] Using dict: {params_dict}\n\n"
            )
            print(log_message)
            with open(log_filename, "a+") as file:
                # Write the log to the file
                file.write(log_message)

def json_to_list_dicts(json_filename):
    # Initialize an empty list
    list_dicts = []
    # Open the JSON file
    # Note: the files are not real JSON files, as they violate JSON fromatting, so we read them as
    #       text files
    with open("./searchable_params/" + json_filename, "r") as json_file:
        # Loop through each line in the file
        for line in json_file:
            # Load the json data from the line
            json_data = json.loads(line)
            # Add the json data to the python dict
            list_dicts.append(json_data)
    return list_dicts

def read_and_start_iteration(thread_number, filename):
    list_dicts = json_to_list_dicts(filename)
    iterate_simulate(thread_number, list_dicts, filename)

if __name__ == "__main__":

    # Check if the user provided an argument
    if len(sys.argv) > 2:
        filename = sys.argv[1]
        idx = sys.argv[1]
        # Do something with the user input
        print("You entered:", filename)
    else:
        print("No argument was provided")
        raise ValueError

    read_and_start_iteration(idx, filename)
