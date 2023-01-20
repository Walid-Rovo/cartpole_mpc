import numpy as np
from casadi import *

class EKF:
    def __init__(self, y):
        # Defining parameters
        g = 9.81 #m/s^2
        mc = 2.4 #kg

        # Timestep
        dt = 0.05

        # nx the number of states, nu the number of inputs and ny the number of measurements
        self.nx = 6
        self.nu = 1
        self.ny = 2
        # constant flow from both pumps
        self.u_k = np.array([0.0]).reshape([-1, 1])
        self.x = SX.sym("x",self.nx)
        self.u = SX.sym("u",self.nu)
        # Discrete system
        den_t1 = mc + self.x[5] - self.x[5]*(cos(self.x[2])**2)
        den_t2 = (self.x[5] * self.x[4] * cos(self.x[2])**2 - (mc + self.x[4]) * self.x[4])
        delta_h = 0.05

        self.x_next_0 = self.x[0] + delta_h*self.x[1]
        self.x_next_1 = self.x[1] + delta_h*(((self.x[5]*self.x[4]* (self.x[3]**2) *sin(self.x[2]))/den_t1) - ((self.x[5]*g*cos(self.x[2])*sin(self.x[2]))/self.en_t1) + (self.u[0]/den_t1))
        self.x_next_2 = self.x[2] + delta_h*self.x[3]
        self.x_next_3 = self.x[3] + delta_h*((((mc+self.x[5])*g*sin(self.x[2]))/den_t2) + ((self.x[5]*self.x[4]* (self.x[3]**2) *sin(self.x[2])*cos(self.x[2]))/den_t2) + ((self.u[0]*cos(self.x[2]))/den_t2))
        self.x_next_4 = self.x[4] + 0
        self.x_next_5 = self.x[5] + 0
        self.x_next = vertcat(self.x_next_0,self.x_next_1,self.x_next_2, self.x_next_3, self.x_next_4, self.x_next_5)

        #Continuous system
        self.x_dot = vertcat(
            self.x[1],
            (self.u + self.x[5] * self.x[4] * sin(self.x[2]) * self.x[3]**2 - self.x[5] * g * cos(self.x[2]) * sin(self.x[2]))
            / (mc + self.x[5] - self.x[5] * cos(self.x[2])**2),
            self.x[3],
            (self.u * cos(self.x[2]) + (mc + self.x[5]) * g * sin(self.x[2]) + self.x[5] * self.x[4] * cos(self.x[2]) * sin(self.x[2]) * self.x[3]**2)
            / (self.x[5] * self.x[4] * cos(self.x[2])**2 - (mc + self.x[4]) * self.x[4]),
            0,
            0
        )
        system_disc = Function('x_next',[self.x, self.u],[self.x_next])
        ode = {'x': self.x, 'ode': self.x_dot, 'p': self.u}
        # By default the solver integrates from 0 to 1. We change the final time to dt.
        opts = {'tf': dt}
        # Create the solver object.
        self.ode_solver = integrator('F', 'idas', ode, opts)
        y_k = vertcat(self.x[0], self.x[2])
        self.measurement = Function('y_k',[self.x],[y_k])
        a_tilde = jacobian(self.x_next,self.x)
        c_tilde = jacobian(y_k,self.x)

        self.A_fun = Function('A_fun', [self.x,self.u], [a_tilde])
        self.C_fun = Function('C_fun', [self.x], [c_tilde])
        self.R = np.diag([1e-4, 1e-4])

# Covariance matrix of the state noise
        q0 = 1e-6
        q1 = 1e-6
        q2 = 1e-6
        q3 = 1e-6
        q4 = 1e-6
        q5 = 1e-6

        self.Q = np.array([[q0,0,0,0,0,0],
                    [0,q1,0,0,0,0],
                    [0,0,q2,0,0,0],
                    [0,0,0,q3,0,0],
                    [0,0,0,0,q4,0],
                    [0,0,0,0,0,q5]])
    
    def discrete_EKF_filter(self, x0, xhat, P_0, wx, wy):
    
        
        # Measurement update / Correction
        C_tilda = self.C_fun(xhat).full()
        L = P_0 @ C_tilda.T @ inv(C_tilda @ P_0 @ C_tilda.T + self.R)

        res_integrator = self.ode_solver(x0=x0, p=self.u_k) 
        x0 = res_integrator['xf']   #plant state
        y = self.measurement(x0+ wx).full() + wy
        xhat = xhat + L @ (y - C_tilda @ xhat) #observed state after correction
        P_0 = (np.eye(self.nx) - L @ C_tilda) @ P_0

        # Prediction_step
        xhat = self.ode_solver(x0=xhat,p=self.u_k)['xf'].full()  # x[k+1|k]   observed state prior correction
        A = self.A_fun(xhat, self.u_k)
        P_0 = A @ P_0 @ A.T + self.Q  # P[k+1|k]
        return xhat