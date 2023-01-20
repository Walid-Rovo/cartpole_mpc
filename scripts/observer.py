import numpy as np
from casadi import *

class EKF:
    def __init__(
        self,
        x0,
        P0=np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
        Q=np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
        R=np.diag([1e-4, 1e-4]),
        dt=0.02
        ):
        # Defining parameters
        g = 9.81  # m/s^2
        mc = 2.4  # kg
        # nx the number of states, nu the number of inputs and ny the number of measurements
        self.xhat = x0
        self.nx = 6
        self.nu = 1
        self.ny = 2
        x = SX.sym("x",self.nx)
        u = SX.sym("u",self.nu)
        # Discrete system
        den_t1 = mc + x[5] - x[5]*(cos(x[2])**2)
        den_t2 = (x[5] * x[4] * cos(x[2])**2 - (mc + x[4]) * x[4])

        x_next = vertcat(
            x[0] + dt*x[1],
            x[1] + dt*(((x[5]*x[4]* (x[3]**2) *sin(x[2]))/den_t1) - ((x[5]*g*cos(x[2])*sin(x[2]))/den_t1) + (u[0]/den_t1)),
            x[2] + dt*x[3],
            x[3] + dt*((((mc+x[5])*g*sin(x[2]))/den_t2) + ((x[5]*x[4]* (x[3]**2) *sin(x[2])*cos(x[2]))/den_t2) + ((u[0]*cos(x[2]))/den_t2)),
            x[4] + 0,
            x[5] + 0,
        )

        #Continuous system
        x_dot = vertcat(
            x[1],
            (u + x[5] * x[4] * sin(x[2]) * x[3]**2 - x[5] * g * cos(x[2]) * sin(x[2]))
            / (mc + x[5] - x[5] * cos(x[2])**2),
            x[3],
            (u * cos(x[2]) + (mc + x[5]) * g * sin(x[2]) + x[5] * x[4] * cos(x[2]) * sin(x[2]) * x[3]**2)
            / (x[5] * x[4] * cos(x[2])**2 - (mc + x[4]) * x[4]),
            0,
            0,
        )
        ode = {'x': x, 'ode': x_dot, 'p': u}
        # By default the solver integrates from 0 to 1. We change the final time to dt.
        opts = {'tf': dt}
        # Create the solver object.
        self.ode_solver = integrator('F', 'idas', ode, opts)
        # C: vertcat below
        y0 = vertcat(x[0], x[2])
        self.measurement = Function('y_k',[x],[y0])
        a_tilde = jacobian(x_next,x)
        c_tilde = jacobian(y0, x)

        self.A_fun = Function('A_fun', [x,u], [a_tilde])
        self.C_fun = Function('C_fun', [x], [c_tilde])
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
        self.xhat = self.xhat + L @ (y - C_tilda @ self.xhat) #observed state after correction
        self.P = (np.eye(self.nx) - L @ C_tilda) @ self.P

        # Prediction_step
        self.xhat = self.ode_solver(x0=self.xhat,p=u)['xf'].full()  # x[k+1|k]   observed state prior correction
        A = self.A_fun(self.xhat, u)
        self.P = A @ self.P @ A.T + self.Q  # P[k+1|k]
        return self.xhat