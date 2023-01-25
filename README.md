# CartPole MPC

A demo of the classic cartpole problem employing model-predictive control with an extended kalman filter state estimator, coded using the powerful CasADi optimization toolbox.

<img src="docs/media/MPC_EKF_tracking_noise_15secs.webp" />

## Running

To run with `pygame` visuals, install the `PAS_sim` environment:
```
cd cartpole_mpc
conda env create -f PAS_environment_game.yml
```
To run a demo, do the following:

```
cd cartpole/scripts
conda activate PAS_sim
python main.py
```

## Features

  - Random parameter search, see `produce_random_params.py` and `search_params.py`
  - Real-time control; the integrators, optimizers were tuned to run fast
  - Nice plots!
## License

Copyright © 2023 Walid-Rovo \
CartPole MPC is made available under the terms of [the MIT License](LICENSE).

## Sources
 - PAS TU-Dortmund for the overall MPC and EKF architecture:
 https://pas.bci.tu-dortmund.de/
 - CasADi for the Python toolbox:
 ```
 @Article{Andersson2019,
  author = {Joel A E Andersson and Joris Gillis and Greg Horn
            and James B Rawlings and Moritz Diehl},
  title = {{CasADi} -- {A} software framework for nonlinear optimization
           and optimal control},
  journal = {Mathematical Programming Computation},
  volume = {11},
  number = {1},
  pages = {1--36},
  year = {2019},
  publisher = {Springer},
  doi = {10.1007/s12532-018-0139-4}
}
```
 - OpenAI Gym (now Farama-Foundation Gymnasium) for the `pygame` visuals
