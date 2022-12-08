"""
Reference: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
"""
import math
import numpy as np

class PendulumOnCart():
    def __init__(self, render=False):
        # physical parameters
        self.g = 9.81
        self.mc = 2.4 
        self.mp = 0.23 
        self.mtotal = self.mc + self.mp
        # self.length = 0.5  # actually half the pole's length
        # self.polemass_length = self.masspole * self.length
        self.lp = 0.36
        self.force_mag = 1.0
        self.tau = 0.02  # seconds between state updates
        # initial state
        self.x0 = 0.5
        self.x_dot0 = 0
        self.theta0 = 3.1
        self.theta_dot0 = 0
        # KF initial estimates
        self.xKF0 = -0.5
        self.x_dotKF0 = 0
        self.thetaKF0 = 2.8
        self.theta_dotKF0 = 0
        self.lpKF = 0.1
        self.mpKF = 0.1
        # states
        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0
        # render
        self.world_width = 2.4 * 2
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.render_bool = render
        self.render_fps = 50

    def step(self, action):
        x, x_dot, theta, theta_dot = (self.x, self.x_dot, self.theta, self.theta_dot,)
        u = self.force_mag * action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        xdd = (u + self.mp*self.lp*sintheta*theta_dot**2-self.mp*self.g*costheta*sintheta)/(self.mc+self.mp-self.mp*costheta**2)
        thetadd = (u*costheta+(self.mc+self.mp)*self.g*sintheta+self.mp*self.lp*costheta*sintheta*theta_dot**2)/(self.mp*self.lp*costheta**2-(self.mc+self.mp)*self.lp)

        self.x = x + self.tau * x_dot
        self.x_dot = x_dot + self.tau * xdd
        self.theta = theta + self.tau * theta_dot
        self.theta_dot = theta_dot + self.tau * thetadd

        if self.render_bool:
            self.render()

    def reset(self):
        # state/observations.
        self.x, self.x_dot, self.theta, self.theta_dot = (0, 0, 0, 0,)

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError(
                "pygame is not installed, run `pip install pygame`"
            )

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.world_width
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.lp)
        cartwidth = 50.0
        cartheight = 30.0

        x = (self.x, self.x_dot, self.theta, self.theta_dot,)

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
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

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

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


if __name__ == '__main__':
    pendulum = PendulumOnCart(render=True)
    # pendulum.masspole = 0.01
    N_SIM_STEPS = 300
    N_FORCE_STEPS = 5
    FORCE_MAGNITUTUDE = 0.1  # Newtons

    # triangle force
    # force_vec = np.linspace(-FORCE_MAGNITUTUDE, FORCE_MAGNITUTUDE, num=N_FORCE_STEPS)
    # force_vec = np.concatenate([force_vec, -force_vec])
    # force_vec = np.repeat(force_vec, repeats=int(N_SIM_STEPS/N_FORCE_STEPS))

    # random normal noise force
    np.random.seed(42)
    force_vec = np.random.normal(loc=0, scale=FORCE_MAGNITUTUDE, size=N_SIM_STEPS)

    pendulum.reset()
    for f in force_vec:
        pendulum.step(action=f)
    pendulum.close()
