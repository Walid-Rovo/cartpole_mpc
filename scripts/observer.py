import numpy as np
from casadi import *
import matplotlib.pyplot as plt


class EKF:
    def __init__(
        self,
        x0,
        P0=np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
        Q=np.diag([1e-4, 1e-4, 1e-2, 1e-1, 1e-4, 1e-4]),
        R=np.diag([1e-4, 1e-4]),
        dt=0.02,
    ):
        # Defining parameters
        g = 9.81  # m/s^2
        mc = 2.4  # kg
        # nx the number of states, nu the number of inputs and ny the number of measurements
        self.xhat = x0
        self.nx = 6
        self.nu = 1
        self.ny = 2
        x = SX.sym("x", self.nx)
        u = SX.sym("u", self.nu)
        # Discrete system
        den_t1 = mc + x[5] - x[5] * (cos(x[2]) ** 2)
        den_t2 = x[5] * x[4] * cos(x[2]) ** 2 - (mc + x[4]) * x[4]

        x_next = vertcat(
            x[0] + dt * x[1],
            x[1]
            + dt
            * (
                ((x[5] * x[4] * (x[3] ** 2) * sin(x[2])) / den_t1)
                - ((x[5] * g * cos(x[2]) * sin(x[2])) / den_t1)
                + (u[0] / den_t1)
            ),
            x[2] + dt * x[3],
            x[3]
            + dt
            * (
                (((mc + x[5]) * g * sin(x[2])) / den_t2)
                + ((x[5] * x[4] * (x[3] ** 2) * sin(x[2]) * cos(x[2])) / den_t2)
                + ((u[0] * cos(x[2])) / den_t2)
            ),
            x[4] + 0,
            x[5] + 0,
        )

        # Continuous system
        x_dot = vertcat(
            x[1],
            (u + x[5] * x[4] * sin(x[2]) * x[3] ** 2 - x[5] * g * cos(x[2]) * sin(x[2]))
            / (mc + x[5] - x[5] * cos(x[2]) ** 2),
            x[3],
            (
                u * cos(x[2])
                + (mc + x[5]) * g * sin(x[2])
                + x[5] * x[4] * cos(x[2]) * sin(x[2]) * x[3] ** 2
            )
            / (x[5] * x[4] * cos(x[2]) ** 2 - (mc + x[4]) * x[4]),
            0,
            0,
        )
        ode = {"x": x, "ode": x_dot, "p": u}
        # By default, the solver integrates from 0 to 1. We change the final time to dt.
        opts = {"tf": dt, "dump_in": True, "dump_out": True}
        # Create the solver object.
        self.ode_solver = integrator("F", "idas", ode, opts)
        # C: vertcat below
        y0 = vertcat(x[0], x[2])
        self.measurement = Function("y_k", [x], [y0])
        a_tilde = jacobian(x_next, x)
        c_tilde = jacobian(y0, x)

        self.A_fun = Function("A_fun", [x, u], [a_tilde])
        self.C_fun = Function("C_fun", [x], [c_tilde])
        self.R = R

        # Covariance matrix of the state noise
        self.Q = Q
        self.P = P0

    def discrete_EKF_filter(self, y, u):
        """
        use as such:
            # measurement: C@x
            y = measurement(x + wx) + wy
            xhat = ekf.discrete_EKF_filter(y)
        """
        C_tilda = self.C_fun(self.xhat).full()
        L = self.P @ C_tilda.T @ inv(C_tilda @ self.P @ C_tilda.T + self.R)
        self.xhat = self.xhat + L @ (y - C_tilda @ self.xhat)  # observed state after correction
        self.P = (np.eye(self.nx) - L @ C_tilda) @ self.P

        # Prediction_step
        self.xhat = self.ode_solver(x0=self.xhat, p=u)[
            "xf"
        ].full()  # x[k+1|k]   observed state prior correction
        A = self.A_fun(self.xhat, u)
        self.P = A @ self.P @ A.T + self.Q  # P[k+1|k]
        return self.xhat

    def discrete_EKF_filter_demo(self, x0, x0_observer, P_0, Q, R, N_sim):
        # defining empty list
        x_data = [x0]
        x_hat_data = [x0_observer]
        y_measured = []
        # defining noise variance
        var_x = Q @ np.ones([self.nx, 1])
        var_y = R @ np.ones([self.ny, 1])

        for j in range(N_sim):
            # Gaussian noise for the plant and measurement
            u_k = np.array([0.0]).reshape([-1, 1])
            wx = np.random.normal(0, np.sqrt(var_x)).reshape(self.nx, 1)
            wy = np.random.normal(0, np.sqrt(var_y)).reshape(self.ny, 1)

            # Measurement update / Correction
            C_tilda = self.C_fun(x0_observer).full()
            L = P_0 @ C_tilda.T @ inv(C_tilda @ P_0 @ C_tilda.T + self.R)

            # x0 = system_cont(x0, u_k).full() #plant state
            res_integrator = self.ode_solver(x0=x0, p=u_k)
            x0 = res_integrator["xf"]  # plant state
            y = self.measurement(x0).full()
            x0_observer = x0_observer + L @ (
                y - C_tilda @ x0_observer
            )  # observed state after correction
            P_0 = (np.eye(self.nx) - L @ C_tilda) @ P_0

            x_data.append(x0)
            x_hat_data.append(x0_observer)
            y_measured.append(y)

            # Prediction_step
            x0_observer = self.ode_solver(x0=x0_observer, p=u_k)[
                "xf"
            ].full()  # x[k+1|k]   observed state prior correction
            A = self.A_fun(x0_observer, u_k)
            P_0 = A @ P_0 @ A.T + Q  # P[k+1|k]

        x_data = np.concatenate(x_data, axis=1)
        x_hat_data = np.concatenate(x_hat_data, axis=1)

        return x_data, x_hat_data, y_measured

    def visualize(self, x_data, x_hat_data):
        fig, ax = plt.subplots(self.nx)
        fig.suptitle("EKF Observer")

        for i in range(self.nx):
            ax[i].plot(x_data[i, :], label="Real State")
            ax[i].plot(x_hat_data[i, :], "r--", label="Estimated State")
            ax[i].set_ylabel("x{}".format(i))
            ax[i].legend(loc="lower right")

        ax[-1].set_xlabel("time_steps")
        plt.show()


if __name__ == "__main__":
    DT = 0.05
    N_sim = int(10 / DT)
    x0 = np.array([0.5, 0, 3.1, 0, 0.36, 0.23]).reshape([-1, 1])
    x0_observer = np.array([-0.5, 0, 1.0, 0, 0.1, 0.1]).reshape([-1, 1])
    observer = EKF(x0=x0, dt=DT)

    # Define the measurement covariance matrix
    R = np.diag([1e-4, 1e-4])

    # Covariance matrix of the state noise
    q0 = 1e-6
    q1 = 1e-6
    q2 = 1e-6
    q3 = 1e-6
    q4 = 1e-6
    q5 = 1e-6

    Q = np.array(
        [
            [q0, 0, 0, 0, 0, 0],
            [0, q1, 0, 0, 0, 0],
            [0, 0, q2, 0, 0, 0],
            [0, 0, 0, q3, 0, 0],
            [0, 0, 0, 0, q4, 0],
            [0, 0, 0, 0, 0, q5],
        ]
    )

    # Covariance Matrix of initial error
    P0 = Q * 200
    N_sim = 500

    # call EKF function
    [plant_state, obs_state_discrete, plant_measurement] = observer.discrete_EKF_filter_demo(
        x0, x0_observer, P0, Q, R, N_sim
    )
    observer.visualize(plant_state, obs_state_discrete)
