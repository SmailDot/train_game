import sys
import math
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
        self.env = env or GameEnv()
        self.agent = agent
        self.mode = "AI" if agent is not None else "Human"
        self.font = pygame.font.SysFont(None, 28)
        self.large_font = pygame.font.SysFont(None, 36)
        self.n = 1
        # current episode score
        self.current_score = 0.0
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

    def draw_playfield(self, state):
        # draw background for play area
        pygame.draw.rect(self.screen, (20, 20, 30), self.play_area)

        # map env coords to pixels
        s_y = state[0]  # normalized y
        y_px = int(s_y * self.HEIGHT)

        # ball
        ball_x = int(self.play_area.width * 0.2)
        ball_y = y_px
        pygame.draw.circle(self.screen, (255, 200, 50), (ball_x, ball_y), 12)

        # obstacle: use normalized x (state[2]) back to px within play_area
        ob_x_norm = state[2]
        ob_x_px = int(self.play_area.right - ob_x_norm * self.play_area.width)
        gap_top = int(state[3] * self.HEIGHT)
        gap_bottom = int(state[4] * self.HEIGHT)

        # draw top and bottom rects
        pygame.draw.rect(self.screen, (10, 120, 10), (ob_x_px, 0, 40, gap_top))
        pygame.draw.rect(self.screen, (10, 120, 10), (ob_x_px, gap_bottom, 40, self.HEIGHT - gap_bottom))

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

        # leaderboard
        lb_top = nn_rect.bottom + 12
        lb_title = self.font.render("Leaderboard:", True, (220, 220, 220))
        self.screen.blit(lb_title, (self.panel.left + 20, lb_top))
        for idx, (name, score) in enumerate(self.leaderboard[:5]):
            t = self.font.render(f"{idx+1}. {name} - {score}", True, (200, 200, 200))
            self.screen.blit(t, (self.panel.left + 20, lb_top + 24 + idx * 22))

    def handle_click(self, pos):
        if self.btn_human.collidepoint(pos):
            self.mode = "Human"
            return
        if self.btn_ai.collidepoint(pos):
            self.mode = "AI"
            return
        if hasattr(self, "btn_export") and self.btn_export.collidepoint(pos):
            self.export_weights()
            return
        if self.btn_board.collidepoint(pos):
            # toggle leaderboard maybe
            return

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

    def run(self):
        s = self.env.reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if self.mode == "Human" and event.key == pygame.K_SPACE:
                        # queue a human jump for the main step
                        self.human_jump = True

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
