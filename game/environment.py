import random
from typing import Optional

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
    ScrollSpeed = 2.2  # 放慢初始速度，方便早期探索
    MaxScrollSpeed = 4.2  # 難度逐步提升的上限
    # how much to increase scroll per passed obstacle (fraction per pass)
    ScrollIncreasePerPass = 0.025  # 每通過一次逐漸加速
    WarmupSteps = 200  # 起始緩速步數
    # spacing between obstacles (minimum distance)
    ObstacleSpacing = 260.0
    ObstacleWidth = 40.0  # 垂直障礙物的寬度（需與 UI 一致）
    CollisionPadding = 4.0  # 額外碰撞緩衝，避免「擦邊通過」的作弊感
    # 勝利條件：達到此分數即通關
    WinningScore = 6666

    # Reward shaping & curriculum knobs
    AlignmentRewardScale = 0.2
    VelocityPenaltyScale = 0.02
    StepPenalty = 0.0
    GapShrinkPerPass = 0.6  # Relaxed from 0.8 to make late game slightly more fair
    MaxGapHalf = 120.0
    MinGapHalf = 70.0

    def __init__(self, seed=None, max_steps=None, frame_skip=4):
        self.rng = random.Random(seed)
        self.obstacles = []  # list of (x, gap_top, gap_bottom, passed) tuples
        # Optional step limit (None for unlimited play)
        self.max_steps = max_steps
        self.ball_radius = 12.0
        self.frame_skip = frame_skip
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
        # number of passed obstacles (used to scale scroll/difficulty)
        self.passed_count = 0
        # Create initial obstacles spaced out from right edge
        spawn_x = self.MaxDist
        for _ in range(3):  # Start with 3 obstacles visible
            gap_center = self.rng.uniform(150.0, self.ScreenHeight - 150.0)
            gap_half = self._sample_gap_half()
            gap_top = gap_center - gap_half
            gap_bottom = gap_center + gap_half
            # (x, gap_top, gap_bottom, passed_flag)
            self.obstacles.append([spawn_x, gap_top, gap_bottom, False])
            spawn_x += self.ObstacleSpacing

        self.t = 0
        self.done = False
        # track cumulative score for the current episode
        self.episode_score = 0.0
        return self._get_state()

    def _sample_gap_half(self) -> float:
        """Sample a gap half-width that shrinks as more obstacles are passed."""
        shrink = min(
            self.MaxGapHalf - self.MinGapHalf,
            self.passed_count * self.GapShrinkPerPass,
        )
        max_half = self.MaxGapHalf - shrink
        min_half = max(self.MinGapHalf, max_half - 20.0)
        return self.rng.uniform(min_half, max_half)

    def apply_difficulty_profile(self, profile: dict) -> None:
        """Dynamically adjust difficulty-related attributes from a profile dict."""

        if not isinstance(profile, dict):
            return

        for key, value in profile.items():
            if not hasattr(self, key):
                continue
            setattr(self, key, value)

    def _find_nearest_obstacle(self):
        """Return the nearest obstacle that is still in front of the player."""
        nearest_obs = None
        min_dist = float("inf")
        for obs in self.obstacles:
            if obs[0] >= -20 and obs[0] < min_dist:
                min_dist = obs[0]
                nearest_obs = obs
        return nearest_obs

    def _find_next_obstacle(self, current_obs):
        """Return the obstacle immediately following the current one."""
        if current_obs is None:
            return None

        next_obs = None
        min_dist = float("inf")
        current_x = current_obs[0]

        for obs in self.obstacles:
            # Find obstacle that is strictly to the right of the current one
            # We use a small epsilon to handle float comparisons if needed
            if obs[0] > current_x + 1.0 and obs[0] < min_dist:
                min_dist = obs[0]
                next_obs = obs
        return next_obs

    def _get_state(self):
        # Find the nearest obstacle ahead of the player (at x ~= 0)
        nearest_obs = self._find_nearest_obstacle()

        # If no obstacle found, use a dummy far away obstacle
        if nearest_obs is None:
            ob_x, gap_top, gap_bottom = self.MaxDist, 250.0, 350.0
            # Next obstacle is also dummy
            next_x, next_gap_top, next_gap_bottom = self.MaxDist * 2, 250.0, 350.0
        else:
            ob_x, gap_top, gap_bottom = nearest_obs[0], nearest_obs[1], nearest_obs[2]

            # Find next obstacle
            next_obs = self._find_next_obstacle(nearest_obs)
            if next_obs is None:
                # If no next obstacle, assume similar to current or default
                next_x = ob_x + self.ObstacleSpacing
                next_gap_top, next_gap_bottom = 250.0, 350.0
            else:
                next_x, next_gap_top, next_gap_bottom = (
                    next_obs[0],
                    next_obs[1],
                    next_obs[2],
                )

        s = np.array(
            [
                self.y / self.ScreenHeight,
                self.vy / self.MaxAbsVel,
                min(ob_x, self.MaxDist) / self.MaxDist,
                gap_top / self.ScreenHeight,
                gap_bottom / self.ScreenHeight,
                (gap_top - self.y) / self.ScreenHeight,  # Relative dist to top
                (gap_bottom - self.y) / self.ScreenHeight,  # Relative dist to bottom
                # --- New Features: Next Obstacle ---
                min(next_x, self.MaxDist * 2)
                / (self.MaxDist * 2),  # Normalize with larger range
                next_gap_top / self.ScreenHeight,
                next_gap_bottom / self.ScreenHeight,
                (next_gap_top - self.y) / self.ScreenHeight,
                (next_gap_bottom - self.y) / self.ScreenHeight,
            ],
            dtype=np.float32,
        )
        return s

    def step(self, action: int):
        """action: 0 or 1 (jump)

        returns: state, reward, done, info
        """
        total_reward = 0.0
        done = False
        info = {}

        # Frame Skip: Repeat action for n frames
        for _ in range(self.frame_skip):
            r, d = self._physics_step(action)
            total_reward += r
            if d:
                done = True
                break
            # Only apply action on the first frame of the skip?
            # Usually in Atari, action is repeated.
            # But here, 'jump' is an impulse.
            # If we repeat jump 4 times, it's 4 impulses.
            # That might be too strong.
            # Strategy: If action is 1 (jump), apply it only on the first frame.
            # If action is 0 (do nothing), apply it on all frames.
            # However, standard FrameSkip repeats the action.
            # Let's stick to standard: repeat the action.
            # But wait, if I hold 'jump' for 4 frames, I get 4 impulses?
            # In Flappy Bird, you tap to jump.
            # So action 1 should probably only happen once.
            # Let's change logic: Action 1 is "Tap", Action 0 is "Fall".
            # If I output 1, I tap once, then fall for the rest of the skip.
            action = 0  # Reset action to 0 after first sub-step to simulate a "Tap"

        return self._get_state(), total_reward, done, info

    def _physics_step(self, action: int):
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
        difficulty_multiplier = 1.0 + self.passed_count * self.ScrollIncreasePerPass
        current_scroll = min(
            self.ScrollSpeed * difficulty_multiplier,
            self.MaxScrollSpeed,
        )

        # gentle warmup phase to ease early exploration
        if self.t < self.WarmupSteps:
            warmup_ratio = self.t / float(self.WarmupSteps)
            current_scroll = (
                self.ScrollSpeed + (current_scroll - self.ScrollSpeed) * warmup_ratio
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
            gap_half = self._sample_gap_half()
            gap_top = gap_center - gap_half
            gap_bottom = gap_center + gap_half
            self.obstacles.append([spawn_x, gap_top, gap_bottom, False])
            break  # Only spawn one per step

        reward = 0.0  # Removed StepPenalty as requested
        done = False
        win = False

        # 球的半徑（與 UI 中的繪製大小一致）
        ball_radius = 12.0
        obstacle_width = float(getattr(self, "ObstacleWidth", 40.0))
        # Remove collision padding to match visuals exactly
        collision_padding = 0.0
        ball_left = -ball_radius
        ball_right = ball_radius

        # reward shaping：保持在間隙中央並抑制劇烈速度
        alignment_score = 0.0
        nearest_obs = self._find_nearest_obstacle()
        if nearest_obs is not None:
            gap_top = nearest_obs[1]
            gap_bottom = nearest_obs[2]
            gap_center = 0.5 * (gap_top + gap_bottom)
            gap_half = 0.5 * (gap_bottom - gap_top)
            distance_from_center = abs(self.y - gap_center)
            normalized_distance = min(
                1.0, distance_from_center / (gap_half + ball_radius)
            )
            # Sharpened alignment reward: Quadratic falloff
            # Encourages being perfectly in the center much more than being "just okay"
            alignment_score = (1.0 - normalized_distance) ** 2
            # Reduced Alignment Reward Scale from 0.2 to 0.05 to avoid reward hacking
            reward += 0.05 * alignment_score

        reward -= self.VelocityPenaltyScale * (abs(self.vy) / self.MaxAbsVel)

        # Check collisions and pass-through events for all obstacles
        for obs in self.obstacles:
            ob_x, gap_top, gap_bottom = obs[0], obs[1], obs[2]

            # Detect pass-through event: obstacle crosses player's x ~= 0
            # Player is at x=0, check if obstacle剛好完全通過
            obs_right_edge = ob_x + obstacle_width
            if (
                not obs[3]
                and obs_right_edge <= 0
                and obs_right_edge > -current_scroll * 2
            ):
                obs[3] = True  # Mark as passed
                # Check if ball was in the gap when passing
                # 使用球體邊緣而不是球心來判斷，與碰撞檢測一致
                ball_top = self.y - ball_radius
                ball_bottom = self.y + ball_radius
                # 球體沒有與障礙物碰撞（即在間隙內通過）
                if ball_top >= gap_top and ball_bottom <= gap_bottom:
                    reward += 25.0  # Increased from 10.0 to prioritize survival/passing
                    self.passed_count += 1

        # Collision detection
        if self.y < 0 or self.y > self.ScreenHeight:
            done = True
            reward -= 5.0

        for obs in self.obstacles:
            ob_x, gap_top, gap_bottom = obs[0], obs[1], obs[2]
            # Simple AABB collision
            # Player is at x=0, width=ball_radius*2
            # Obstacle is at ob_x, width=ObstacleWidth

            # Check horizontal overlap
            # Player x range: [-ball_radius, ball_radius]
            # Obstacle x range: [ob_x, ob_x + obstacle_width]

            if (ball_right > ob_x + collision_padding) and (
                ball_left < ob_x + obstacle_width - collision_padding
            ):
                # Check vertical overlap (collision with top or bottom pipe)
                ball_top = self.y - ball_radius
                ball_bottom = self.y + ball_radius

                if (ball_top < gap_top + collision_padding) or (
                    ball_bottom > gap_bottom - collision_padding
                ):
                    done = True
                    reward -= 5.0
                    break

        # Check win condition
        if self.episode_score + reward >= self.WinningScore:
            win = True
            done = True
            reward += 1000.0

        self.episode_score += reward
        self.t += 1
        if self.max_steps and self.t >= self.max_steps:
            done = True

        return reward, done
        if self.max_steps is not None and self.t >= self.max_steps:
            done = True

        self.done = done
        info = {
            "episode_score": float(self.episode_score),
            "passed_count": self.passed_count,
            "scroll_speed": current_scroll,
            "alignment_score": alignment_score,
            "win": win,
            "alive_time": self.t,
        }
        if done:
            info.setdefault("win", win)
            info["episode"] = {
                "r": float(self.episode_score),
                "l": self.t,
                "passed": self.passed_count,
            }
            # reset episode score on done; caller will call reset()
            self.episode_score = 0.0
        return self._get_state(), float(reward), bool(done), info

    def render_frame(
        self, width: int = 640, height: Optional[int] = None
    ) -> np.ndarray:
        """Render a simple RGB array representing the current game state."""

        height = int(height or self.ScreenHeight)
        width = int(max(width, 320))
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # base colors
        bg_color = np.array([18, 22, 33], dtype=np.uint8)
        obstacle_color = np.array([80, 110, 165], dtype=np.uint8)
        gap_glow = np.array([120, 180, 230], dtype=np.uint8)
        ball_color = np.array([255, 215, 120], dtype=np.uint8)
        win_color = np.array([140, 220, 120], dtype=np.uint8)

        frame[:] = bg_color

        world_min = -80.0
        world_max = self.MaxDist + self.ObstacleSpacing * 2.0
        world_span = max(world_max - world_min, 1.0)

        def world_to_px(x_val: float) -> int:
            norm = (x_val - world_min) / world_span
            px = int(norm * width)
            return int(np.clip(px, 0, width - 1))

        # draw gap glows and obstacles
        for ob_x, gap_top, gap_bottom, _ in self.obstacles:
            x0 = world_to_px(ob_x)
            x1 = world_to_px(ob_x + self.ObstacleWidth)
            if x1 <= x0:
                x1 = min(width - 1, x0 + 1)

            top = max(0, int(np.floor(gap_top)))
            bottom = min(height, int(np.ceil(gap_bottom)))

            if top > 0:
                frame[0:top, x0:x1] = obstacle_color
            if bottom < height:
                frame[bottom:height, x0:x1] = obstacle_color

            # glow inside the gap for readability
            glow_region = slice(max(top - 4, 0), min(bottom + 4, height))
            frame[glow_region, x0:x1] = (
                0.8 * frame[glow_region, x0:x1] + 0.2 * gap_glow
            ).astype(np.uint8)

        # draw player ball
        cy = int(np.clip(self.y, 0, height - 1))
        cx = world_to_px(0.0)
        radius = int(getattr(self, "ball_radius", 12) or 12)
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        frame[mask] = ball_color

        # progress bar for winning score
        progress = float(np.clip(self.episode_score / self.WinningScore, 0.0, 1.0))
        bar_x0, bar_y0 = 20, 20
        bar_w = width - 40
        bar_h = 12
        filled = max(1, int(bar_w * progress))
        frame[bar_y0 : bar_y0 + bar_h, bar_x0 : bar_x0 + filled] = win_color
        frame[bar_y0 : bar_y0 + bar_h, bar_x0 + filled : bar_x0 + bar_w] = np.array(
            [40, 55, 70], dtype=np.uint8
        )

        # overlay passed obstacle count as tiny ticks
        tick_count = min(self.passed_count, 50)
        if tick_count > 0:
            tick_spacing = bar_w / float(tick_count)
            for idx in range(tick_count):
                x = int(bar_x0 + idx * tick_spacing)
                frame[bar_y0 + bar_h : bar_y0 + bar_h + 4, x : x + 2] = win_color

        return frame

    def render(self, mode: Optional[str] = None):
        mode = mode or "human"
        if mode == "rgb_array":
            return self.render_frame()

        if mode == "human":
            try:
                import pygame
            except ImportError:
                pass
            else:
                if not hasattr(self, "window"):
                    pygame.init()
                    self.window = pygame.display.set_mode((640, 480))
                    pygame.display.set_caption(f"Training Env {id(self)}")
                    self.clock = pygame.time.Clock()

                frame = self.render_frame(width=640, height=480)
                # Frame is (H, W, 3) RGB. Pygame expects (W, H, 3) for make_surface
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

                self.window.blit(surface, (0, 0))
                pygame.display.flip()

                # Handle events to prevent "Not Responding"
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                return

        # Minimal text render (for smoke testing)
        # obs_count = len(self.obstacles)

    def close(self):
        """Clean up resources."""
        if hasattr(self, "window"):
            try:
                import pygame

                pygame.display.quit()
                pygame.quit()
            except ImportError:
                pass
            del self.window

        # Minimal text render (for smoke testing)
        # obs_count = len(self.obstacles)
        # nearest_x = min((obs[0] for obs in self.obstacles), default=999.9)
        # print(
        #     f"t={self.t} y={self.y:.1f} vy={self.vy:.1f} obstacles={obs_count} "
        #     f"nearest_x={nearest_x:.1f}"
        # )

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
