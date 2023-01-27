import datetime
import random
import json
import os

TOTAL_PERMS = 50
STEP_STD = 1e2
FILEPATH = "./searchable_params/params_perms-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"

PARAMS_DICT = {
    "seed": [
        0,
    ],
    "N_time": [
        30,
    ],
    "dt": [
        0.04,
    ],
    # EKF
    "EKF_P0_0": [
        1e-6,
    ],
    "EKF_P0_1": [
        1e-6,
    ],
    "EKF_P0_2": [
        1e-6,
    ],
    "EKF_P0_3": [
        1e-6,
    ],
    "EKF_P0_4": [
        1e-6,
    ],
    "EKF_P0_5": [
        1e-6,
    ],
    "EKF_Q_0": [
        1e-6,
    ],
    "EKF_Q_1": [
        1e-6,
    ],
    "EKF_Q_2": [
        1e-6,
    ],
    "EKF_Q_3": [
        1e-6,
    ],
    "EKF_Q_4": [
        1e-3,
    ],
    "EKF_Q_5": [
        1e-3,
    ],
    "EKF_R_0": [
        1e-6,
    ],
    "EKF_R_1": [
        1e-6,
    ],
    # x measurement error std
    "wy_0": [
        1e-6,
    ],
    # theta measurement error std
    "wy_2": [
        1e-6,
    ],
    # MPC
    "max_solver_iter": [
        40,
        500,
        2000,
    ],
    "MPC_K": [
        2,
    ],
    "MPC_N": [
        40,
        60,
        80,
    ],
    "MPC_Q_0": [
        1e-1,
        1e-2,
        1e-3,
    ],
    "MPC_Q_1": [
        1e-6,
    ],
    "MPC_Q_2": [
        1e-2,
        1e-3,
        1e-4,
    ],
    "MPC_Q_3": [
        1e-1,
        1e-3,
        1e-5,
    ],
    "MPC_R": [
        1e-3,
        1e-4,
        1e-5,
    ],
    "MPC_x_bound": [
        10.0,
    ],
    "MPC_xdot_bound": [
        20.0,
        30.0,
        40.0,
    ],
    "MPC_theta_bound": [
        0.8 * 3.1415,
        1.0 * 3.1415,
        1.2 * 3.1415,
        1.4 * 3.1415,
    ],  #
    "MPC_thetadot_bound": [
        0.3,
        0.5,
        0.7,
        0.9,
    ],  #
    "MPC_u_bound": [
        25,
    ],
}


def append_record(record, filepath):
    with open(filepath, 'a+') as f:
        json.dump(record, f)
        f.write(os.linesep)


def shuffle_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    random.shuffle(lines)
    with open(file_path, 'w') as file:
        file.writelines(lines)


def random_boolean_from_probability(probability):
    return random.random() < probability


if __name__ == "__main__":

    list_of_keys = list(PARAMS_DICT.keys())
    list_of_lengths = [len(PARAMS_DICT[key]) for key in list_of_keys]

    product_number = 1
    for length in list_of_lengths:
        product_number *= length

    mean_index_step = product_number / TOTAL_PERMS
    print(f"Parameter combinations search space: {product_number: e}\n"
          f"Selecting: {TOTAL_PERMS: e}\n"
          f"Stepping: {mean_index_step} with std. {STEP_STD}")

    index_list = [0]
    while index_list[-1] < product_number:
        index_list.append(index_list[-1] + int(random.gauss(mean_index_step, STEP_STD)))
    if index_list[-1] > product_number:
        index_list.pop()
    print(index_list[:5], index_list[-6:-1])

    for n in index_list:
        index = n
        index_list = []
        for length in reversed(list_of_lengths):
            index_list.insert(0, index % length)
            index = index // length
        keys_with_values = {}
        for j, key in enumerate(list_of_keys):
            keys_with_values[key] = PARAMS_DICT[key][index_list[j]]
        keys_with_values["seed"] = random.randint(1, 10000000)
        append_record(keys_with_values, FILEPATH)
        shuffle_file(FILEPATH)
