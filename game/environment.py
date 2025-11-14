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
    MaxDist = 150  # 減少初始距離，讓障礙物更快到達
    MaxAbsVel = 20.0
    # horizontal scroll speed (pixels per step) base value
    ScrollSpeed = 3.0  # 增加滾動速度
    # how much to increase scroll per passed obstacle (fraction per pass)
    ScrollIncreasePerPass = 0.01  # 降低速度增長率：從 0.02 改為 0.01，避免過快
    # spacing between obstacles (minimum distance)
    ObstacleSpacing = 250.0
    # 勝利條件：達到此分數即通關
    WinningScore = 6666

    def __init__(self, seed=None, max_steps=None):
        self.rng = random.Random(seed)
        self.obstacles = []  # list of (x, gap_top, gap_bottom, passed) tuples
        # Optional step limit (None for unlimited play)
        self.max_steps = max_steps
        self.reset()

    def reset(self):  # noqa: C901
        # ball: spawn at a random vertical position in the middle 60% of screen
        # avoid spawning too close to top or bottom edges
        safe_min = self.ScreenHeight * 0.4  # 40% from top (更居中)
        safe_max = self.ScreenHeight * 0.6  # 60% from top (更居中)
        self.y = self.rng.uniform(safe_min, safe_max)
        self.vy = 0.0

        # Initialize multiple obstacles for smooth scrolling
        self.obstacles = []
        # Create initial obstacles spaced out from right edge
        spawn_x = self.MaxDist
        for _ in range(3):  # Start with 3 obstacles visible
            gap_center = self.rng.uniform(150.0, self.ScreenHeight - 150.0)
            gap_half = self.rng.uniform(80.0, 100.0)  # 增加間隙大小 160-200
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
        min_dist = float("inf")
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
        current_scroll = self.ScrollSpeed * (
            1.0 + self.passed_count * self.ScrollIncreasePerPass
        )

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

            gap_center = self.rng.uniform(150.0, self.ScreenHeight - 150.0)
            gap_half = self.rng.uniform(80.0, 100.0)  # 增加間隙大小 160-200
            gap_top = gap_center - gap_half
            gap_bottom = gap_center + gap_half
            self.obstacles.append([spawn_x, gap_top, gap_bottom, False])
            break  # Only spawn one per step

        reward = 0.1  # 每步存活獎勵，鼓勵存活更久
        done = False

        # 球的半徑（與 UI 中的繪製大小一致）
        ball_radius = 12.0

        # 碰撞窗口：只有當球心進入障礙物範圍內才檢測碰撞
        # 原本 collision_window = 14，現在改為 0，表示球心必須與障礙物重疊
        collision_window = ball_radius  # 球心在障礙物的 x 範圍內

        # Check collisions and pass-through events for all obstacles
        for obs in self.obstacles:
            ob_x, gap_top, gap_bottom = obs[0], obs[1], obs[2]

            # Detect pass-through event: obstacle crosses player's x ~= 0
            # Player is at x=0, check if obstacle just passed
            if not obs[3] and ob_x <= 0 and ob_x > -current_scroll * 2:
                obs[3] = True  # Mark as passed
                # Check if ball was in the gap when passing
                # 使用球體邊緣而不是球心來判斷，與碰撞檢測一致
                ball_top = self.y - ball_radius
                ball_bottom = self.y + ball_radius
                # 球體沒有與障礙物碰撞（即在間隙內通過）
                if ball_top >= gap_top and ball_bottom <= gap_bottom:
                    reward += 5.0
                    self.passed_count += 1

            # Check collision: obstacle overlaps with player position (x ~= 0)
            # 只有當球心完全進入障礙物的 x 範圍內才判定碰撞
            if -collision_window < ob_x < collision_window:
                # Check if ball is outside the gap
                # 球心超出間隙範圍，且球體與障礙物實際接觸
                ball_top = self.y - ball_radius
                ball_bottom = self.y + ball_radius

                # 檢查球體是否與障礙物頂部或底部碰撞
                # 球的頂部碰到障礙物底部（gap_top 以上的部分）
                # 或球的底部碰到障礙物頂部（gap_bottom 以下的部分）
                if ball_top < gap_top or ball_bottom > gap_bottom:
                    reward -= 5.0
                    done = True
                    break

        # collision / out-of-bounds (top/bottom)
        # hitting ceiling or floor is failure - 球體完全碰到邊界
        if self.y - ball_radius < 0 or self.y + ball_radius > self.ScreenHeight:
            reward -= 5.0
            done = True

        # accumulate episode score
        self.episode_score += float(reward)

        # 檢查勝利條件：達到 99999 分即通關
        if self.episode_score >= self.WinningScore:
            reward += 1000.0  # 給予巨大獎勵
            done = True
            info = {
                "episode_score": float(self.episode_score),
                "win": True,  # 標記為勝利
            }
            self.episode_score = 0.0
            return self._get_state(), float(reward), bool(done), info

        self.t += 1
        if self.max_steps is not None and self.t >= self.max_steps:
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
        """Return obstacles as (x, gap_top, gap_bottom) tuples for rendering."""
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
            min_dist = float("inf")
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
        nearest = min(
            self.obstacles, key=lambda obs: obs[0] if obs[0] >= -20 else float("inf")
        )
        return nearest[1]

    @property
    def gap_bottom(self):
        """Get gap_bottom of nearest obstacle (for backward compatibility)."""
        if not self.obstacles:
            return 350.0
        nearest = min(
            self.obstacles, key=lambda obs: obs[0] if obs[0] >= -20 else float("inf")
        )
        return nearest[2]
