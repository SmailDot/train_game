import random

import numpy as np


class GameEnv:
    """A minimal, fast toy environment matching the spec for smoke-testing.

    State (unnormalized): [y, vy, x_obs, y_gap_top, y_gap_bottom]
    All values are returned normalized by the public constants when step/reset
    so agents can consume normalized inputs.
    
    Note: State only exposes the nearest obstacle for compatibility, but internally
    maintains multiple obstacles for smooth scrolling gameplay.
    """

    ScreenHeight = 600  # 增加遊玩畫面高度以匹配更大的顯示區域
    MaxDist = 300
    MaxAbsVel = 20.0
    # horizontal scroll speed (pixels per step) base value
    ScrollSpeed = 2.0
    # how much to increase scroll per passed obstacle (fraction per pass)
    ScrollIncreasePerPass = 0.02
    # spacing between obstacles (minimum distance)
    ObstacleSpacing = 250.0

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.obstacles = []  # list of (x, gap_top, gap_bottom, passed) tuples
        self.reset()

    def reset(self):  # noqa: C901
        # ball: spawn at a random vertical position in the middle 60% of screen
        # avoid spawning too close to top or bottom edges
        safe_min = self.ScreenHeight * 0.3  # 30% from top
        safe_max = self.ScreenHeight * 0.7  # 70% from top (30% from bottom)
        self.y = self.rng.uniform(safe_min, safe_max)
        self.vy = 0.0
        
        # Initialize multiple obstacles for smooth scrolling
        self.obstacles = []
        # Create initial obstacles spaced out from right edge
        spawn_x = self.MaxDist
        for _ in range(3):  # Start with 3 obstacles visible
            gap_center = self.rng.uniform(80.0, self.ScreenHeight - 80.0)
            gap_half = self.rng.uniform(45.0, 60.0)
            gap_top = gap_center - gap_half
            gap_bottom = gap_center + gap_half
            # (x, gap_top, gap_bottom, passed_flag)
            self.obstacles.append([spawn_x, gap_top, gap_bottom, False])
            spawn_x += self.ObstacleSpacing
        
        self.t = 0
        self.done = False
        # track cumulative score for the current episode
        self.episode_score = 0.0
        # number of passed obstacles (used to scale scroll/difficulty)
        self.passed_count = 0
        return self._get_state()

    def _get_state(self):
        # Find the nearest obstacle ahead of the player (at x ~= 0)
        nearest_obs = None
        min_dist = float('inf')
        for obs in self.obstacles:
            if obs[0] >= -20:  # Only consider obstacles that haven't passed yet
                if obs[0] < min_dist:
                    min_dist = obs[0]
                    nearest_obs = obs
        
        # If no obstacle found, use a dummy far away obstacle
        if nearest_obs is None:
            ob_x, gap_top, gap_bottom = self.MaxDist, 250.0, 350.0
        else:
            ob_x, gap_top, gap_bottom = nearest_obs[0], nearest_obs[1], nearest_obs[2]
        
        s = np.array(
            [
                self.y / self.ScreenHeight,
                self.vy / self.MaxAbsVel,
                min(ob_x, self.MaxDist) / self.MaxDist,
                gap_top / self.ScreenHeight,
                gap_bottom / self.ScreenHeight,
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
        
        # Move all obstacles left by scroll speed
        for obs in self.obstacles:
            obs[0] -= current_scroll
        
        # Remove obstacles that have scrolled off the left side
        self.obstacles = [obs for obs in self.obstacles if obs[0] > -100]
        
        # Spawn new obstacles from the right when needed
        # Keep spawning if rightmost obstacle is too close
        while True:
            if not self.obstacles:
                # No obstacles, spawn one at MaxDist
                spawn_x = self.MaxDist
            else:
                rightmost_x = max(obs[0] for obs in self.obstacles)
                if rightmost_x < self.MaxDist + self.ObstacleSpacing:
                    spawn_x = rightmost_x + self.ObstacleSpacing
                else:
                    break  # No need to spawn yet
            
            gap_center = self.rng.uniform(80.0, self.ScreenHeight - 80.0)
            gap_half = self.rng.uniform(45.0, 60.0)
            gap_top = gap_center - gap_half
            gap_bottom = gap_center + gap_half
            self.obstacles.append([spawn_x, gap_top, gap_bottom, False])
            break  # Only spawn one per step

        reward = -0.1  # time penalty
        done = False
        ball_margin = 15.0  # 球半徑 12 + 小容差

        # Check collisions and pass-through events for all obstacles
        for obs in self.obstacles:
            ob_x, gap_top, gap_bottom = obs[0], obs[1], obs[2]
            
            # Detect pass-through event: obstacle crosses player's x ~= 0
            # Player is at x=0, check if obstacle just passed
            if not obs[3] and ob_x <= 0 and ob_x > -current_scroll * 2:
                obs[3] = True  # Mark as passed
                # Check if ball was in the gap when passing
                if gap_top + ball_margin < self.y < gap_bottom - ball_margin:
                    reward += 5.0
                    self.passed_count += 1
            
            # Check collision: obstacle overlaps with player position (x ~= 0)
            # Player occupies x range roughly [-20, 20]
            if -20 < ob_x < 20:
                # Check if ball is outside the gap
                if not (gap_top + ball_margin < self.y < gap_bottom - ball_margin):
                    reward -= 5.0
                    done = True
                    break

        # collision / out-of-bounds (top/bottom)
        # hitting ceiling or floor is failure
        if self.y < 0 or self.y > self.ScreenHeight:
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
        obs_count = len(self.obstacles)
        nearest_x = min((obs[0] for obs in self.obstacles), default=999.9)
        print(
            f"t={self.t} y={self.y:.1f} vy={self.vy:.1f} obstacles={obs_count} "
            f"nearest_x={nearest_x:.1f}"
        )
    
    def get_all_obstacles(self):
        """Return all obstacles for rendering. Each obstacle is (x, gap_top, gap_bottom)."""
        return [(obs[0], obs[1], obs[2]) for obs in self.obstacles]
    
    # Backward compatibility properties for legacy tests
    @property
    def ob_x(self):
        """Get x position of nearest obstacle (for backward compatibility)."""
        if not self.obstacles:
            return self.MaxDist
        return min(obs[0] for obs in self.obstacles if obs[0] >= -20)
    
    @ob_x.setter
    def ob_x(self, value):
        """Set x position of nearest obstacle (for backward compatibility)."""
        if self.obstacles:
            # Find and update the nearest obstacle
            nearest_idx = 0
            min_dist = float('inf')
            for i, obs in enumerate(self.obstacles):
                if obs[0] < min_dist:
                    min_dist = obs[0]
                    nearest_idx = i
            self.obstacles[nearest_idx][0] = value
    
    @property
    def gap_top(self):
        """Get gap_top of nearest obstacle (for backward compatibility)."""
        if not self.obstacles:
            return 250.0
        nearest = min(self.obstacles, key=lambda obs: obs[0] if obs[0] >= -20 else float('inf'))
        return nearest[1]
    
    @property
    def gap_bottom(self):
        """Get gap_bottom of nearest obstacle (for backward compatibility)."""
        if not self.obstacles:
            return 350.0
        nearest = min(self.obstacles, key=lambda obs: obs[0] if obs[0] >= -20 else float('inf'))
        return nearest[2]
