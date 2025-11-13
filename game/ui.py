import sys
import math
import json
import os
import threading
from typing import Optional

import pygame

from game.environment import GameEnv
from agents.ppo_agent import PPOAgent


class GameUI:
    WIDTH = 1280
    HEIGHT = 720
    BG_COLOR = (30, 30, 40)
    FPS = 60

    def __init__(self, env: Optional[GameEnv] = None, agent: Optional[PPOAgent] = None):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Train Game (1280x720)")
        self.clock = pygame.time.Clock()

        # environment and agent
        self.env = env or GameEnv()
        self.agent = agent

        # start in menu mode; user must choose Human or AI to start a run
        self.mode = "Menu"
        self.selected_mode = None
        self.running = False

        # fonts and counters
        self.font = pygame.font.SysFont(None, 28)
        self.large_font = pygame.font.SysFont(None, 36)
        self.n = 1

        # current episode score
        self.current_score = 0.0
        # latest numeric metrics reported by trainer (thread-safe)
        self.latest_metrics = {}
        # lock for thread-safe updates from trainer thread
        self._lock = threading.Lock()
        # human jump flag (avoid double-stepping in key handler)
        self.human_jump = False

        # UI layout
        self.play_area = pygame.Rect(0, 0, int(self.WIDTH * 0.75), self.HEIGHT)
        self.panel = pygame.Rect(self.play_area.right, 0, self.WIDTH - self.play_area.right, self.HEIGHT)

        # buttons inside panel
        self.btn_human = pygame.Rect(self.panel.left + 20, 40, self.panel.width - 40, 50)
        self.btn_ai = pygame.Rect(self.panel.left + 20, 110, self.panel.width - 40, 50)
        self.btn_board = pygame.Rect(self.panel.left + 20, 180, self.panel.width - 40, 50)

        # leaderboard: list of (name, score), newest entries appended; keep top scores
        self.leaderboard = [("AgentA", 10), ("AgentB", 7), ("Human", 3)]
        # try load persisted leaderboard (if present)
        try:
            self._ensure_checkpoints()
            self._load_scores()
        except Exception:
            pass

        # loss history storage for visualization: dict of name -> list[float]
        self.loss_history = {"policy": [], "value": [], "entropy": [], "total": []}
        # surface for small loss plot
        self.loss_surf_size = (self.panel.width - 40, 120)
        self.loss_surf = pygame.Surface(self.loss_surf_size)

        # thread reference if trainer started
        self.trainer_thread = None
        self._trainer_stop_event = None

    def draw_playfield(self, state):
        # draw background for play area
        pygame.draw.rect(self.screen, (20, 20, 30), self.play_area)

        # map env coords to pixels
        # state[0] is normalized y (0..1), map to play_area height
        # Note: env uses ScreenHeight=200, but UI may have different height
        s_y = state[0]  # normalized y [0,1]
        y_px = int(s_y * self.env.ScreenHeight)  # map back to env's coordinate system
        # clamp to visible range
        y_px = max(10, min(y_px, self.env.ScreenHeight - 10))

        # ball: fixed x position at 20% of play area width
        ball_x = int(self.play_area.width * 0.2)
        ball_y = y_px
        pygame.draw.circle(self.screen, (255, 200, 50), (ball_x, ball_y), 12)

        # obstacle: state[2] is normalized obstacle x [0,1] where 1=MaxDist, 0=at player
        # We need to map this to screen coordinates
        # When ob_x=MaxDist (far right), normalized=1, should appear at right edge of play area
        # When ob_x=0 (at player), normalized=0, should appear at ball_x position
        ob_x_norm = state[2]  # 0..1
        # Map: normalized 0 -> ball_x, normalized 1 -> play_area.width
        ob_x_px = int(ball_x + ob_x_norm * (self.play_area.width - ball_x))
        
        # gap coordinates: already normalized to [0,1] by env
        gap_top_px = int(state[3] * self.env.ScreenHeight)
        gap_bottom_px = int(state[4] * self.env.ScreenHeight)

        # draw obstacle: top pillar and bottom pillar with gap in between
        obstacle_width = 40
        pygame.draw.rect(self.screen, (10, 120, 10), (ob_x_px, 0, obstacle_width, gap_top_px))
        pygame.draw.rect(self.screen, (10, 120, 10), (ob_x_px, gap_bottom_px, obstacle_width, self.env.ScreenHeight - gap_bottom_px))

    def draw_panel(self):
        pygame.draw.rect(self.screen, (18, 18, 22), self.panel)

        # buttons
        pygame.draw.rect(self.screen, (70, 70, 80), self.btn_human)
        pygame.draw.rect(self.screen, (70, 70, 80), self.btn_ai)
        pygame.draw.rect(self.screen, (70, 70, 80), self.btn_board)

        h_text = self.large_font.render("Human Play", True, (240, 240, 240))
        ai_text = self.large_font.render("AI Play", True, (240, 240, 240))
        b_text = self.large_font.render("Leaderboard", True, (240, 240, 240))

        self.screen.blit(h_text, (self.btn_human.left + 10, self.btn_human.top + 10))
        self.screen.blit(ai_text, (self.btn_ai.left + 10, self.btn_ai.top + 10))
        self.screen.blit(b_text, (self.btn_board.left + 10, self.btn_board.top + 10))

        # mode indicator & n
        mode_text = self.font.render(f"Mode: {self.mode}", True, (200, 200, 200))
        n_text = self.font.render(f"目前訓練回合 n={self.n}", True, (200, 200, 200))
        self.screen.blit(mode_text, (self.panel.left + 20, 260))
        self.screen.blit(n_text, (self.panel.left + 20, 290))
        # current score
        score_text = self.font.render(f"本局分數: {int(self.current_score)}", True, (200, 200, 200))
        self.screen.blit(score_text, (self.panel.left + 20, 320))

        # NN placeholder (simple rectangle with title)
        nn_rect = pygame.Rect(self.panel.left + 20, 340, self.panel.width - 40, 200)
        pygame.draw.rect(self.screen, (50, 50, 60), nn_rect)
        nn_title = self.font.render("Neural Net (weights heatmap placeholder)", True, (200, 200, 200))
        self.screen.blit(nn_title, (nn_rect.left + 8, nn_rect.top + 8))

        # try to draw a real weight heatmap if agent/net provides it, otherwise fallback to a fake heatmap
        w = None
        try:
            if self.agent is not None and hasattr(self.agent, "net") and hasattr(self.agent.net, "get_weight_matrix"):
                w = self.agent.net.get_weight_matrix()
            elif hasattr(self.env, "net") and hasattr(self.env.net, "get_weight_matrix"):
                w = self.env.net.get_weight_matrix()
        except Exception:
            w = None

        # prepare grid dims
        grid_rows = 4
        grid_cols = 8
        cell_w = (nn_rect.width - 16) // grid_cols
        cell_h = (nn_rect.height - 40) // grid_rows

        if w is not None:
            try:
                import numpy as _np

                arr = _np.array(w)
                # If 1D, expand to 2D
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

                # Resize or pad/crop to grid_rows x grid_cols
                # Simple approach: scale arr to (grid_rows, grid_cols) via averaging/padding
                # If arr is smaller, pad with zeros; if larger, crop
                r, c = arr.shape
                target = _np.zeros((grid_rows, grid_cols), dtype=_np.float32)
                rr = min(r, grid_rows)
                cc = min(c, grid_cols)
                target[:rr, :cc] = arr[:rr, :cc]

                # normalize to 0..255
                mn = target.min()
                mx = target.max()
                if mx - mn == 0:
                    norm = _np.zeros_like(target)
                else:
                    norm = (target - mn) / (mx - mn)

                for i in range(grid_rows):
                    for j in range(grid_cols):
                        v = float(norm[i, j])
                        cval = int(40 + v * 215)
                        pygame.draw.rect(
                            self.screen,
                            (cval, int(120 * (1 - v)), 200 - cval // 3),
                            (
                                nn_rect.left + 8 + j * cell_w,
                                nn_rect.top + 36 + i * cell_h,
                                cell_w - 2,
                                cell_h - 2,
                            ),
                        )
            except Exception:
                w = None

        if w is None:
            # fallback fake heatmap (animated)
            for i in range(grid_rows):
                for j in range(grid_cols):
                    v = (math.sin(i * 0.5 + j * 0.3 + self.n * 0.05) + 1) / 2
                    c = int(80 + v * 160)
                    pygame.draw.rect(
                        self.screen,
                        (c, 80, 200 - c // 2),
                        (
                            nn_rect.left + 8 + j * cell_w,
                            nn_rect.top + 36 + i * cell_h,
                            cell_w - 2,
                            cell_h - 2,
                        ),
                    )

        # Export weights button
        self.btn_export = pygame.Rect(self.panel.left + 20, nn_rect.top - 40, self.panel.width - 40, 28)
        pygame.draw.rect(self.screen, (80, 60, 80), self.btn_export)
        ex_text = self.font.render("Export weights to TensorBoard", True, (240, 240, 240))
        self.screen.blit(ex_text, (self.btn_export.left + 8, self.btn_export.top + 4))

        # Loss visualization area (below nn_rect)
        loss_rect = pygame.Rect(self.panel.left + 20, nn_rect.bottom + 8, self.panel.width - 40, 120)
        pygame.draw.rect(self.screen, (30, 30, 36), loss_rect)
        loss_title = self.font.render("Losses (policy / value / entropy / total)", True, (200, 200, 200))
        self.screen.blit(loss_title, (loss_rect.left + 8, loss_rect.top + 6))

        # If we're in Menu mode (not running), draw a simple start hint in play area
        if not self.running:
            title = self.large_font.render("Train Game", True, (240, 240, 240))
            hint = self.font.render("Click 'Human Play' or 'AI Play' to start", True, (200, 200, 200))
            # Center in play area
            title_x = self.play_area.left + (self.play_area.width // 2 - title.get_width() // 2)
            hint_x = self.play_area.left + (self.play_area.width // 2 - hint.get_width() // 2)
            self.screen.blit(title, (title_x, 100))
            self.screen.blit(hint, (hint_x, 150))

        # show numeric latest loss values (if any)
        with self._lock:
            lm = dict(self.latest_metrics) if self.latest_metrics else {}

        txt_y = loss_rect.top + 28
        try:
            pol = f"policy: {lm.get('policy_loss'):.4f}" if lm.get('policy_loss') is not None else "policy: -"
            val = f"value: {lm.get('value_loss'):.4f}" if lm.get('value_loss') is not None else "value: -"
            ent = f"entropy: {lm.get('entropy'):.4f}" if lm.get('entropy') is not None else "entropy: -"
            tot = f"total: {lm.get('loss'):.4f}" if lm.get('loss') is not None else "total: -"
        except Exception:
            pol = val = ent = tot = "-"

        self.screen.blit(self.font.render(pol, True, (200, 180, 180)), (loss_rect.left + 8, txt_y))
        self.screen.blit(self.font.render(val, True, (180, 220, 180)), (loss_rect.left + 8 + 140, txt_y))
        self.screen.blit(self.font.render(ent, True, (180, 180, 220)), (loss_rect.left + 8 + 280, txt_y))
        self.screen.blit(self.font.render(tot, True, (220, 220, 140)), (loss_rect.left + 8 + 420, txt_y))

        # draw simple multi-series plot (below numeric summary)
        self._draw_loss_plot(loss_rect.left + 8, loss_rect.top + 48, loss_rect.width - 16, loss_rect.height - 60)

        # leaderboard
        lb_top = nn_rect.bottom + 12
        lb_title = self.font.render("Leaderboard:", True, (220, 220, 220))
        self.screen.blit(lb_title, (self.panel.left + 20, lb_top))
        for idx, (name, score) in enumerate(self.leaderboard[:5]):
            t = self.font.render(f"{idx+1}. {name} - {score}", True, (200, 200, 200))
            self.screen.blit(t, (self.panel.left + 20, lb_top + 24 + idx * 22))

    def handle_click(self, pos):
        # If not running, these buttons start a run
        if not self.running and self.btn_human.collidepoint(pos):
            self.selected_mode = "Human"
            self.mode = "Human"
            self.running = True
            self.agent = None
            self.current_score = 0.0
            # Reset environment and return the new state
            return self.env.reset()
        if not self.running and self.btn_ai.collidepoint(pos):
            self.selected_mode = "AI"
            self.mode = "AI"
            self.running = True
            self.current_score = 0.0
            # ensure agent exists (try to instantiate fallback if missing)
            if self.agent is None:
                try:
                    self.agent = PPOAgent()
                except Exception:
                    self.agent = None
            # Reset environment and return the new state
            return self.env.reset()
        if hasattr(self, "btn_export") and self.btn_export.collidepoint(pos):
            self.export_weights()
            return
        if self.btn_board.collidepoint(pos):
            # toggle leaderboard maybe
            return

    def _draw_loss_plot(self, x, y, w, h):
        """Draw multiple loss series (policy, value, entropy, total) into panel area.

        x,y are screen coordinates, w,h are dimensions.
        """
        surf = pygame.Surface((w, h))
        surf.fill((20, 20, 30))
        # draw dark background
        pygame.draw.rect(surf, (24, 24, 28), (0, 0, w, h))

        # determine max length among series (thread-safe copy)
        with self._lock:
            lh_copy = {k: list(v) for k, v in self.loss_history.items()}

        max_len = 0
        for v in lh_copy.values():
            if v:
                max_len = max(max_len, len(v))

        if max_len < 2:
            # draw a hint text
            hint = self.font.render("No loss data yet", True, (150, 150, 150))
            surf.blit(hint, (6, 6))
            self.screen.blit(surf, (x, y))
            return

        N = min(max_len, w)
        series_colors = {
            "policy": (200, 80, 80),
            "value": (80, 200, 120),
            "entropy": (120, 120, 200),
            "total": (220, 220, 80),
        }

        for name, color in series_colors.items():
            seq = list(lh_copy.get(name, []))
            if not seq:
                continue
            seq = seq[-N:]
            mx = max(seq)
            mn = min(seq)
            denom = mx - mn if mx != mn else 1.0
            points = []
            for i, val in enumerate(seq):
                vx = int(i * (w - 2) / max(1, N - 1))
                vy = int((h - 2) - (val - mn) / denom * (h - 4))
                points.append((vx, vy))
            if len(points) > 1:
                pygame.draw.lines(surf, color, False, points, 2)

        # blit the plot surface
        self.screen.blit(surf, (x, y))

    def export_weights(self):
        """Export actor weights to TensorBoard (if available) or save a local numpy file.

        This is best-effort: if the agent has a network exposing get_weight_matrix(), use it.
        """
        try:
            w = None
            if self.agent is not None and hasattr(self.agent, "net") and hasattr(self.agent.net, "get_weight_matrix"):
                w = self.agent.net.get_weight_matrix()
            elif hasattr(self.env, "net") and hasattr(self.env.net, "get_weight_matrix"):
                w = self.env.net.get_weight_matrix()

            if w is None:
                # nothing to export
                return

            # try tensorboard
            try:
                from torch.utils.tensorboard import SummaryWriter
                import numpy as _np

                writer = SummaryWriter(log_dir="checkpoints/tb_ui")
                # make image: normalize weights to [0,255]
                arr = _np.array(w)
                arr = arr - arr.min()
                denom = arr.max() if arr.max() != 0 else 1.0
                img = (arr / denom * 255).astype(_np.uint8)
                # ensure 3 channels
                if img.ndim == 2:
                    img = _np.stack([img, img, img], axis=2)
                img = img.transpose(2, 0, 1)
                writer.add_image("actor_weights", img, global_step=self.n)
                writer.flush()
                writer.close()
            except Exception:
                # fallback: save numpy
                import numpy as _np

                _np.save(f"checkpoints/weights_n{self.n}.npy", _np.array(w))
        except Exception:
            return

    # --- persistence helpers for leaderboard ---
    def _ensure_checkpoints(self):
        os.makedirs("checkpoints", exist_ok=True)

    def _load_scores(self):
        p = os.path.join("checkpoints", "scores.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # expect list of [name,score]
                if isinstance(data, list):
                    self.leaderboard = [tuple(x) for x in data]
            except Exception:
                # ignore malformed
                pass

    def _save_scores(self):
        p = os.path.join("checkpoints", "scores.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.leaderboard, f, ensure_ascii=False, indent=2)

    # --- trainer/metrics API ---
    def update_losses(self, metrics: dict):
        """Called from trainer (background thread). Accepts a dict of metrics.

        Expected keys: 'it','loss','policy_loss','value_loss','entropy','timesteps','mean_reward','episode_count'
        """
        if not isinstance(metrics, dict):
            return
        with self._lock:
            # keep simple series, append floats if present
            try:
                if metrics.get("policy_loss") is not None:
                    self.loss_history.setdefault("policy", []).append(float(metrics.get("policy_loss")))
                if metrics.get("value_loss") is not None:
                    self.loss_history.setdefault("value", []).append(float(metrics.get("value_loss")))
                if metrics.get("entropy") is not None:
                    self.loss_history.setdefault("entropy", []).append(float(metrics.get("entropy")))
                if metrics.get("loss") is not None:
                    self.loss_history.setdefault("total", []).append(float(metrics.get("loss")))

                # store latest metrics for numeric display
                self.latest_metrics.update({k: metrics.get(k) for k in ("it", "loss", "policy_loss", "value_loss", "entropy", "timesteps", "mean_reward", "episode_count")})
                # also update n (iteration) if present
                try:
                    if metrics.get("it") is not None:
                        self.n = int(metrics.get("it"))
                except Exception:
                    pass
            except Exception:
                # keep training robust to odd metric values
                pass

    def start_trainer(self, trainer, **train_kwargs):
        """Start trainer.train(...) in a background daemon thread and wire metrics to UI.update_losses.

        Example:
            ui.start_trainer(trainer, total_timesteps=5000, env=env)
        """
        if self.trainer_thread is not None and self.trainer_thread.is_alive():
            # already running
            return

        # create a stop event and run trainer in a non-daemon thread so we can join
        stop_event = threading.Event()
        self._trainer_stop_event = stop_event

        def _runner():
            try:
                trainer.train(metrics_callback=self.update_losses, stop_event=stop_event, **train_kwargs)
            except Exception:
                # swallow to avoid killing the UI thread
                pass

        t = threading.Thread(target=_runner, daemon=False)
        t.start()
        self.trainer_thread = t

    def stop_trainer(self, wait=True, timeout=None):
        """Signal the background trainer to stop and optionally join the thread."""
        if self._trainer_stop_event is None:
            return
        try:
            self._trainer_stop_event.set()
            if wait and self.trainer_thread is not None:
                self.trainer_thread.join(timeout)
        finally:
            self._trainer_stop_event = None
            self.trainer_thread = None

    def run(self):
        s = self.env.reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    new_state = self.handle_click(event.pos)
                    if new_state is not None:
                        s = new_state
                elif event.type == pygame.KEYDOWN:
                    if self.running and self.mode == "Human" and event.key == pygame.K_SPACE:
                        # queue a human jump for the main step
                        self.human_jump = True
            
            # if not running (menu mode), only render and wait for user to click start
            if not self.running:
                # render only - don't step the environment
                self.screen.fill(self.BG_COLOR)
                self.draw_playfield(s)
                self.draw_panel()
                pygame.display.flip()
                self.clock.tick(self.FPS)
                continue

            if self.mode == "AI":
                if self.agent is not None:
                    a, _, _ = self.agent.act(s)
                    s, r, done, _ = self.env.step(a)
                else:
                    # no agent: step without action
                    s, r, done, _ = self.env.step(0)
            else:
                # Human mode: perform queued jump or step with 0
                if self.human_jump:
                    action = 1
                    self.human_jump = False
                else:
                    action = 0
                s, r, done, _ = self.env.step(action)

            # update current score and leaderboard when episodes end
            try:
                self.current_score += float(r)
            except Exception:
                pass

            if done:
                name = "AI" if self.mode == "AI" else "Human"
                # record final score (integer)
                self.leaderboard.append((name, int(self.current_score)))
                # keep top 10 entries sorted by score desc
                self.leaderboard = sorted(self.leaderboard, key=lambda x: x[1], reverse=True)[:10]
                # persist leaderboard
                try:
                    self._save_scores()
                except Exception:
                    pass
                # reset current score for next episode
                self.current_score = 0.0

            if done:
                self.n += 1
                s = self.env.reset()

            # render
            self.screen.fill(self.BG_COLOR)
            self.draw_playfield(s)
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(self.FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    # quick run demo using the fallback/random agent if torch not installed
    agent = None
    try:
        agent = PPOAgent()
    except Exception:
        agent = None

    ui = GameUI(agent=agent)
    ui.run()
