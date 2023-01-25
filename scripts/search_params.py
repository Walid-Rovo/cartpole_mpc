import json
import sys
import numpy as np

from simulate import simulate


def iterate_simulate(thread_number, list_dicts, filename):
    log_filename = "param_search/search_results/result-" + filename + ".log"
    lowest_norm = np.inf
    for idx, params_dict in enumerate(list_dicts):
        try:
            res_x_mpc, _, _, _, _ = simulate(params_dict, render=True)
        except (TimeoutError, RuntimeError) as e:
            print(f"[{thread_number}] Iteration {idx} errored: {str(e)}\n\n")
            continue
        norm = np.linalg.norm(np.mean(res_x_mpc, axis=0))
        if norm < lowest_norm:
            lowest_norm = norm
            log_message = (
                f"[{thread_number}] Iteration {idx}/{len(list_dicts)}\n"
                f"[{thread_number}] Reached a new lowest norm: {lowest_norm}\n"
                f"[{thread_number}] Final state: \n{res_x_mpc[-1, :]}\n"
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
    # Note: the files are not real JSON files, as they violate JSON fromatting,
    #       so we read them as text files
    with open("param_search/searchable_params/" + json_filename, "r") as json_file:
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
