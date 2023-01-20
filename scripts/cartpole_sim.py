"""
Reference: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
"""
import math
import numpy as np
from casadi import *

class PendulumOnCart:
    # TODO: figure out why using theta=3.1 put the pole in the down position. It should be up
    def __init__(self, initial_states=[0.0, 0.0, 0.1, 0.0],dt=0.02, render=False):
        # physical parameters
        self.g = -9.81
        self.mc = 2.4
        self.mp = 0.23
        self.mtotal = self.mc + self.mp
        self.dt = dt

        # self.length = 0.5  # actually half the pole's length
        # self.polemass_length = self.masspole * self.length

        self.lp = 0.36
        self.force_mag = 1.0
        self.tau = 0.02  # seconds between state updates

        # # states
        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0

        # CasADi function instantiation
        # x[0] = x, x[1] = x_dot, x[2] = theta, x[3] = theta_dot
        self.nx = 4
        self.nu = 1
        self.x_sym = SX.sym("x", self.nx, 1)
        self.u_sym = SX.sym("u", self.nu, 1)

        self.xdot_sym = vertcat(
            # x[0]_dot =
            self.x_sym[1],
            # self.x_sym[1]_dot =
            (self.u_sym + self.mp * self.lp * sin(self.x_sym[2]) * self.x_sym[3]**2 - self.mp * self.g * cos(self.x_sym[2]) * sin(self.x_sym[2]))
            / (self.mc + self.mp - self.mp * cos(self.x_sym[2])**2),
            # self.x_sym[2]_dot =
            self.x_sym[3],
            # self.x_sym[3]_dot =
            (self.u_sym * cos(self.x_sym[2]) + (self.mc + self.mp) * self.g * sin(self.x_sym[2]) + self.mp * self.lp * cos(self.x_sym[2]) * sin(self.x_sym[2]) * self.x_sym[3]**2)
            / (self.mp * self.lp * cos(self.x_sym[2])**2 - (self.mc + self.lp) * self.lp),
        )

        self.system = Function("system", [self.x_sym, self.u_sym], [self.xdot_sym])
        # CasADi integrator instantiation
        self.x_sym0 = np.array(initial_states).reshape(self.nx,1)
        # The CasADi integrator needs a dictionary of the states ('x'), the system state-space eqself.ations as a CasADi symbolic expression ('ode'), input ('p').
        self.ode = {'x': self.x_sym, 'ode': self.xdot_sym, 'p': self.u_sym}

        # By default the solver integrates from 0 to 1. We change the final time to dt.
        self.opts = {'tf': self.dt}

        # Create the solver object.
        # https://casadi.sourceforge.net/api/html/db/d3d/classcasadi_1_1Integrator.html#:~:text=Constructor%20%26%20Destructor%20Documentation
        #                      name, solv typ, function dict, options dict
        self.ode_solver = integrator('F', 'idas', self.ode, self.opts)

        # render
        self.world_width = 2.4 * 2
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.render_bool = render
        self.render_fps = 50

    def step(self, action: float):
        action = np.array([[action]])
        res_integrator = self.ode_solver(x0=self.x_sym0, p=action)
        # integrator returns dict, one of the values is the final x
        self.x_sym0 = res_integrator['xf']

        self.x = self.x_sym0[0]
        self.x_dot = self.x_sym0[1]
        self.theta = self.x_sym0[2]
        self.theta_dot = self.x_sym0[3]

        if self.render_bool:
            self.render()

        state = np.array[self.x.full(), self.x_dot.full(), self.theta.full(), self.theta_dot.full()].reshape(-1,1)

        return state

    def reset(self):
        # state/observations.
        self.x, self.x_dot, self.theta, self.theta_dot = (
            0,
            0,
            0,
            0,
        )

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError("pygame is not installed, run `pip install pygame`")

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.world_width
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.lp)
        cartwidth = 50.0
        cartheight = 30.0

        x = (
            self.x,
            self.x_dot,
            self.theta,
            self.theta_dot,
        )

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))


        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )


        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        pygame.event.pump()
        self.clock.tick(self.render_fps)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == "__main__":

    pendulum = PendulumOnCart(render=True)

    # pendulum.masspole = 0.01
    N_SIM_STEPS = 1000
    N_FORCE_STEPS = 5
    FORCE_MAGNITUDE = 0.1  # Newtons
    N_FORCE_STEPS = 50

    # triangle force
    # force_vec = np.linspace(-FORCE_MAGNITUDE, FORCE_MAGNITUDE, num=N_FORCE_STEPS)
    # force_vec = np.concatenate([force_vec, -force_vec])
    # force_vec = np.repeat(force_vec, repeats=int(N_SIM_STEPS/N_FORCE_STEPS))

    # random normal noise force
    np.random.seed(42)
    force_vec = np.zeros(N_SIM_STEPS)
    force_vec[:N_FORCE_STEPS] = np.random.normal(
        loc=0, scale=FORCE_MAGNITUDE, size=N_FORCE_STEPS
    )

    pendulum.reset()
    for f in force_vec:
        pendulum.step(action=f)

    pendulum.close()
