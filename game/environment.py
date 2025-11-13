import random

import numpy as np


class GameEnv:
    """A minimal, fast toy environment matching the spec for smoke-testing.

    State (unnormalized): [y, vy, x_obs, y_gap_top, y_gap_bottom]
    All values are returned normalized by the public constants when step/reset
    so agents can consume normalized inputs.
    """

    ScreenHeight = 200
    MaxDist = 300
    MaxAbsVel = 20.0

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):  # noqa: C901
        import random



        class GameEnv:
            """A minimal, fast toy environment matching the spec for smoke-testing.

            State (unnormalized): [y, vy, x_obs, y_gap_top, y_gap_bottom]
            All values are returned normalized by the public constants when step/reset
            so agents can consume normalized inputs.
            """

            ScreenHeight = 200
            MaxDist = 300
            MaxAbsVel = 20.0

            def __init__(self, seed=None):
                self.rng = random.Random(seed)
                self.reset()

            def reset(self):
                # ball
                self.y = self.ScreenHeight * 0.5
                self.vy = 0.0
                # one obstacle ahead
                self.ob_x = 220.0
                gap_center = self.ScreenHeight * 0.5
                gap_half = 20.0
                self.gap_top = gap_center - gap_half
                self.gap_bottom = gap_center + gap_half
                self.t = 0
                self.done = False
                return self._get_state()

            def _get_state(self):
                s = np.array(
                    [
                        self.y / self.ScreenHeight,
                        self.vy / self.MaxAbsVel,
                        min(self.ob_x, self.MaxDist) / self.MaxDist,
                        self.gap_top / self.ScreenHeight,
                        self.gap_bottom / self.ScreenHeight,
                    ],
                    dtype=np.float32,
                )
                return s

            def step(self, action: int):
                """action: 0 or 1 (jump)

                returns: state, reward, done, info
                """
                # physics params
                gravity = 0.6
                jump_impulse = -7.0
                dt = 1.0

                # apply action
                if action:
                    self.vy = jump_impulse

                # integrate
                self.vy += gravity * dt
                self.vy = max(min(self.vy, self.MaxAbsVel), -self.MaxAbsVel)
                self.y += self.vy * dt

                # move obstacle left
                self.ob_x -= 2.0
                if self.ob_x < -50:
                    # spawn new obstacle with random gap
                    self.ob_x = 220.0
                    gap_center = self.rng.uniform(40, self.ScreenHeight - 40)
                    gap_half = self.rng.uniform(16, 32)
                    self.gap_top = gap_center - gap_half
                    self.gap_bottom = gap_center + gap_half

                reward = -0.1  # time penalty
                done = False

                # check pass through: when obstacle passes x ~ 0 and center crosses
                if (
                    0 < self.ob_x < 2.0
                    and abs(self.y - ((self.gap_top + self.gap_bottom) / 2)) < 30
                ):
                    reward += 10.0

                # collision
                # assume ball radius = 5
                if self.y < 0 or self.y > self.ScreenHeight:
                    reward -= 100.0
                    done = True
                else:
                    # collision when obstacle near player and ball outside gap
                    if (
                        self.ob_x < 10
                        and not (self.gap_top + 5 < self.y < self.gap_bottom - 5)
                    ):
                        reward -= 100.0
                        done = True

                self.t += 1
                if self.t > 1000:
                    done = True

                self.done = done
                return self._get_state(), float(reward), bool(done), {}

            def render(self):
                # Minimal text render (for smoke testing)
                print(
                    f"t={self.t} y={self.y:.1f} vy={self.vy:.1f} obx={self.ob_x:.1f} "
                    f"gap=[{self.gap_top:.1f},{self.gap_bottom:.1f}]"
                )
