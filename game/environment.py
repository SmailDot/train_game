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
    # horizontal scroll speed (pixels per step) base value
    ScrollSpeed = 2.0
    # how much to increase scroll per passed obstacle (fraction per pass)
    ScrollIncreasePerPass = 0.02

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):  # noqa: C901
        # ball: spawn at a random vertical position (not too close to edges)
        self.y = self.rng.uniform(20.0, float(self.ScreenHeight - 20.0))
        self.vy = 0.0
        # one obstacle ahead (spawn at semi-random distance)
        self.ob_x = self.rng.uniform(180.0, float(self.MaxDist))
        gap_center = self.ScreenHeight * 0.5
        gap_half = 20.0
        self.gap_top = gap_center - gap_half
        self.gap_bottom = gap_center + gap_half
        self.t = 0
        self.done = False
        # track previous obstacle x to detect pass events
        self.prev_ob_x = self.ob_x
        # track cumulative score for the current episode
        self.episode_score = 0.0
        # number of passed obstacles (used to scale scroll/difficulty)
        self.passed_count = 0
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

        # compute dynamic scroll speed that increases with passed obstacles
        current_scroll = self.ScrollSpeed * (1.0 + self.passed_count * self.ScrollIncreasePerPass)
        # move obstacle left by current scroll speed
        self.prev_ob_x = self.ob_x
        self.ob_x -= current_scroll
        if self.ob_x < -50:
            # spawn new obstacle at a random distance ahead with random gap
            self.ob_x = self.rng.uniform(180.0, float(self.MaxDist))
            gap_center = self.rng.uniform(40.0, self.ScreenHeight - 40.0)
            gap_half = self.rng.uniform(16.0, 32.0)
            self.gap_top = gap_center - gap_half
            self.gap_bottom = gap_center + gap_half

        reward = -0.1  # time penalty
        done = False

        # detect pass-through event: obstacle crosses player's x ~= 0
        # If previous ob_x > 0 and current ob_x <= 0, the obstacle passed the player
        gap_center = (self.gap_top + self.gap_bottom) / 2
        # award +5 for passing through if ball is within gap
        if self.prev_ob_x > 0 and self.ob_x <= 0:
            if self.gap_top + 5 < self.y < self.gap_bottom - 5:
                reward += 5.0
                # increase passed counter (difficulty)
                try:
                    self.passed_count += 1
                except Exception:
                    self.passed_count = 0

        # collision / out-of-bounds (top/bottom)
        # hitting ceiling or floor is failure
        if self.y < 0 or self.y > self.ScreenHeight:
            reward -= 5.0
            done = True
        else:
            # collision when obstacle near player and ball outside gap
            if self.ob_x < 10 and not (self.gap_top + 5 < self.y < self.gap_bottom - 5):
                reward -= 5.0
                done = True

        # accumulate episode score
        self.episode_score += float(reward)

        self.t += 1
        if self.t > 1000:
            done = True

        self.done = done
        info = {"episode_score": float(self.episode_score)}
        if done:
            # reset episode score on done; caller will typically call reset()
            # but provide final score in info
            self.episode_score = 0.0
        return self._get_state(), float(reward), bool(done), info

    def render(self):
        # Minimal text render (for smoke testing)
        print(
            f"t={self.t} y={self.y:.1f} vy={self.vy:.1f} obx={self.ob_x:.1f} "
            f"gap=[{self.gap_top:.1f},{self.gap_bottom:.1f}]"
        )
