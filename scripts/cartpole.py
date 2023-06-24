import aerosandbox as asb
import aerosandbox.numpy as np

import pygame

class CartPole:
    def __init__(
        self,
        dt=0.02,
        render=True,
        ) -> None:
        self.dt = dt
        # Initial conditions
        self.g = -9.81
        self.mc = 2.4
        self.mp = 0.23
        self.mtotal = self.mc + self.mp

        # self.length = 0.5  # actually half the pole's length
        # self.polemass_length = self.masspole * self.length

        self.lp = 0.36
        self.force_mag = 1.0
        self.tau = 0.02  # seconds between state updates

        self.render_bool = render
        self.isopen = True
        self.screen = None
        self.clock = None
        self.quit_flag = False

    def ode(self, X, U):
        X_next = np.array([
            # dx =
            [X[1]],
            # ddx =
            [(U + self.mp * self.lp * np.sin(X[2]) * X[3] ** 2
              - self.mp * self.g * np.cos(X[2]) * np.sin(X[2]))
             / (self.mc + self.mp - self.mp * np.cos(X[2]) ** 2)],
            # dtheta =
            [X[3]],
            # ddtheta =
            [(
                U * np.cos(X[2])
                + (self.mc + self.mp) * self.g * np.sin(X[2])
                + self.mp * self.lp * np.cos(X[2]) * np.sin(X[2]) * X[3] ** 2
            ) / (self.mp * self.lp * np.cos(X[2]) ** 2 - (self.mc + self.lp) * self.lp)],
        ])
        if len(X_next.shape) > 2:
            X_next = np.squeeze(X_next, axis=2)
        return X_next

    def objective_fun(self, X, U, Q, R):
        return X.T @ Q @ X + U.T @ R @ U

    def integrate_Euler(self, X, U, dt):
        X_p = X + dt * self.ode(X, U)
        return X_p

    def integrate_RK4(self, X, U, dt):
        k1 = self.ode(X, U)
        k2 = self.ode(X + k1 * dt / 2, U)
        k3 = self.ode(X + k2 * dt / 2, U)
        k4 = self.ode(X + k3 * dt, U)
        return X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def ocp(self, x_0, x_f, Horiz, x_lims, u_lims):
        # opti = cas.Opti()

        # X = opti.variable(4, 11)
        # U = opti.variable(1, 10)

        nx = 4
        nu = 1

        def ix(i): return range(i * nx, (i + 1) * nx)  # noqa: E704

        opti = asb.Opti()

        X = opti.variable(np.zeros(nx * (Horiz + 1)))
        U = opti.variable(np.zeros(nu * Horiz))

        # U_lb = opti.parameter()
        # U_ub = opti.parameter()

        # opti.subject_to(-3.0 >= U[0])

        # Initial condition
        opti.subject_to(X[ix(0)] == x_0)

        # System dynamics constraints and cost function
        J = 0
        R = 0.0001
        Q = np.diag(np.array([0.01, 0.01, 1.0, 0.01]))
        for i in range(Horiz):
            opti.subject_to(X[ix(i + 1)] == self.integrate_RK4(X[ix(i)], U[i], self.dt))
            # opti.subject_to(x_lims[0] <= X[ix(i)] <= np.tile(x_lims[1], Horiz + 1))
            # opti.subject_to(U_lb <= U <= U_ub)
            opti.subject_to(-50.0 <= U[i])
            opti.subject_to(U[i] <= 50.0)
            J += self.objective_fun(X[ix(i + 1)], U[i], Q, R)

        Qf = np.diag(np.array([0.01, 0.01, 0.01, 0.01]))
        J += X[ix(Horiz)].T @ Qf @ X[ix(Horiz)]

        # Terminal state constraint
        # opti.subject_to(X[ix(Horiz)] == x_f)

        # Path constraints
        # opti.set_value(U_lb, u_lims[0])
        # opti.set_value(U_ub, u_lims[1])
        # opti.subject_to(np.tile(x_lims[0], Horiz + 1) <= X <= np.tile(x_lims[1], Horiz + 1))
        # opti.subject_to(np.tile(u_lims[0], Horiz + 1) <= U <= np.tile(u_lims[1], Horiz + 1))

        # Solver and solving
        opti.minimize(J)
        opti.solver('ipopt')
        soln = opti.solve(verbose=False, max_iter=20)

        return soln.value(X), soln.value(U)

    def forward_step(self, X, U):
        X = self.integrate_RK4(X, U, self.dt)

        if self.render_bool:
            self.render(X)
        return X

    # def reset(self):
    #     # state/observations.
    #     X = np.zeros_like(X)

    def render(self, X):
        # render
        world_width = 10  # 4.8
        screen_width = 1600  # 600
        screen_height = 480  # 400
        render_fps = int(1 / self.dt)
        # player_action = 0.0
        # pact_step = 0.5
        # pact_max = 1.5
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            print("pygame import error, disabling rendering")
            self.render_bool = False
            return

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        keys = pygame.key.get_pressed()
        # if keys[pygame.K_LEFT]:
        #     player_action -= pact_step
        # elif keys[pygame.K_RIGHT]:
        #     player_action += pact_step
        # else:
        #     player_action = 0.0
        # if player_action > pact_max:
        #     player_action = pact_max
        # elif player_action < -pact_max:
        #     player_action = -pact_max

        if keys[pygame.K_q]:
            self.quit_flag = True
            self.close()

        world_width = 40
        scale = screen_width / world_width
        polewidth = 7.0
        polelen = scale * (4 * self.lp)
        cartwidth = polelen * 2 *  0.4
        cartheight = polelen * 2 * 0.2

        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = X[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.hline(surf, 0, screen_width, carty, (0, 0, 0))
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-X[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101))

        try:
            gfxdraw.aacircle(
                surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )
            gfxdraw.filled_circle(
                surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )
        except OverflowError:
            print(cartx, carty)

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        pygame.event.pump()
        self.clock.tick(render_fps)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.quit()
            # pygame.display.quit()
            self.isopen = False


if __name__ == "__main__":
    # Initial conditions
    x_0 = np.array([[0.0],
                    [0.0],
                    [np.deg2rad(5.0)],
                    [0.0]])

    x_f = np.zeros_like(x_0)

    x_lims = (np.ones(4, dtype="float32") * -3.14,
              np.ones(4, dtype="float32") * 3.14,)
    u_lims = (-3.0, 3.0)

    # Init CartPole with init. conditions
    cartpole = CartPole(render=True)

    counter = 0
    try:
        while True:
            _, u = cartpole.ocp(x_0, x_f, 5, x_lims, u_lims)
            x_0 = cartpole.forward_step(x_0, u[0])
            print(u[0], x_0[2])

            # x_0 = cartpole.forward_step(x_0, 4.5492890003726085)
    except KeyboardInterrupt:
        raise SystemExit from KeyboardInterrupt
