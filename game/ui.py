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

        # fonts and counters - 使用支持中文的字體
        # 嘗試使用系統中文字體，如果找不到則使用 pygame 默認字體
        chinese_fonts = ['microsoftyahei', 'microsoftyaheimicrosoftyaheiui', 'microsoftyaheiui', 
                        'simhei', 'simsun', 'kaiti', 'fangsong', 'nsimsun', 
                        'msgothic', 'mspgothic', 'notosanscjk', 'notosanscjksc',
                        'arial', 'verdana']  # fallback to common fonts
        
        self.font = None
        self.large_font = None
        
        # 嘗試載入中文字體
        for font_name in chinese_fonts:
            try:
                self.font = pygame.font.SysFont(font_name, 28)
                self.large_font = pygame.font.SysFont(font_name, 36)
                # 測試是否能正確渲染中文
                test_surface = self.font.render("測試", True, (255, 255, 255))
                if test_surface.get_width() > 0:  # 如果能渲染出內容
                    break
            except Exception:
                continue
        
        # 如果還是沒有找到合適的字體，使用 pygame 默認字體
        if self.font is None:
            self.font = pygame.font.Font(None, 28)
            self.large_font = pygame.font.Font(None, 36)
        
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
        
        # Game state flags
        self.paused = False  # ESC 暫停狀態
        self.game_over = False  # 遊戲結束狀態
        self.show_pause_menu = False  # 顯示暫停選單

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
        s_y = state[0]  # normalized y [0,1]
        y_px = int(s_y * self.env.ScreenHeight)  # map back to env's coordinate system
        # clamp to visible range
        y_px = max(10, min(y_px, self.env.ScreenHeight - 10))

        # ball: fixed x position at 20% of play area width
        ball_x = int(self.play_area.width * 0.2)
        ball_y = y_px
        
        # Draw all obstacles (scrolling from right to left)
        obstacle_width = 40
        if hasattr(self.env, 'get_all_obstacles'):
            for ob_x, gap_top, gap_bottom in self.env.get_all_obstacles():
                # Map obstacle x from env coordinates to screen coordinates
                # env: x=0 is at player (ball_x), x=MaxDist is far right (play_area.width)
                # Linear mapping: screen_x = ball_x + (ob_x / MaxDist) * (play_area.width - ball_x)
                scale = (self.play_area.width - ball_x) / self.env.MaxDist
                ob_x_px = int(ball_x + ob_x * scale)
                
                # Only draw obstacles that are visible on screen
                if -obstacle_width < ob_x_px < self.play_area.width:
                    gap_top_px = int(gap_top)
                    gap_bottom_px = int(gap_bottom)
                    
                    # draw top pillar and bottom pillar with gap in between
                    pygame.draw.rect(self.screen, (10, 120, 10), 
                                   (ob_x_px, 0, obstacle_width, gap_top_px))
                    pygame.draw.rect(self.screen, (10, 120, 10), 
                                   (ob_x_px, gap_bottom_px, obstacle_width, 
                                    self.env.ScreenHeight - gap_bottom_px))
        
        # Draw ball on top of obstacles
        pygame.draw.circle(self.screen, (255, 200, 50), (ball_x, ball_y), 12)

    def draw_panel(self):
        pygame.draw.rect(self.screen, (18, 18, 22), self.panel)

        # buttons
        pygame.draw.rect(self.screen, (70, 70, 80), self.btn_human)
        pygame.draw.rect(self.screen, (70, 70, 80), self.btn_ai)
        pygame.draw.rect(self.screen, (70, 70, 80), self.btn_board)

        h_text = self.large_font.render("人類遊玩", True, (240, 240, 240))
        ai_text = self.large_font.render("AI 遊玩", True, (240, 240, 240))
        b_text = self.large_font.render("排行榜", True, (240, 240, 240))

        self.screen.blit(h_text, (self.btn_human.left + 10, self.btn_human.top + 10))
        self.screen.blit(ai_text, (self.btn_ai.left + 10, self.btn_ai.top + 10))
        self.screen.blit(b_text, (self.btn_board.left + 10, self.btn_board.top + 10))

        # mode indicator & n
        mode_name = "人類" if self.mode == "Human" else ("AI" if self.mode == "AI" else "選單")
        mode_text = self.font.render(f"模式: {mode_name}", True, (200, 200, 200))
        n_text = self.font.render(f"訓練回合: {self.n}", True, (200, 200, 200))
        self.screen.blit(mode_text, (self.panel.left + 20, 260))
        self.screen.blit(n_text, (self.panel.left + 20, 285))
        # current score
        score_text = self.font.render(f"本局分數: {int(self.current_score)}", True, (200, 200, 200))
        self.screen.blit(score_text, (self.panel.left + 20, 310))

        # NN placeholder (simple rectangle with title)
        nn_rect = pygame.Rect(self.panel.left + 20, 345, self.panel.width - 40, 160)
        pygame.draw.rect(self.screen, (50, 50, 60), nn_rect)
        nn_title = self.font.render("神經網路權重熱力圖", True, (200, 200, 200))
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

        # Loss visualization area (below nn_rect) - 調整位置避免重疊
        loss_rect = pygame.Rect(self.panel.left + 20, nn_rect.bottom + 10, self.panel.width - 40, 100)
        pygame.draw.rect(self.screen, (30, 30, 36), loss_rect)
        loss_title = self.font.render("訓練損失 (策略/價值/熵/總計)", True, (200, 200, 200))
        self.screen.blit(loss_title, (loss_rect.left + 8, loss_rect.top + 6))

        # If we're in Menu mode (not running), draw a simple start hint in play area
        if not self.running:
            title = self.large_font.render("訓練遊戲", True, (240, 240, 240))
            hint = self.font.render("點擊「人類遊玩」或「AI 遊玩」開始", True, (200, 200, 200))
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
            pol = f"策略: {lm.get('policy_loss'):.4f}" if lm.get('policy_loss') is not None else "策略: -"
            val = f"價值: {lm.get('value_loss'):.4f}" if lm.get('value_loss') is not None else "價值: -"
            ent = f"熵: {lm.get('entropy'):.4f}" if lm.get('entropy') is not None else "熵: -"
            tot = f"總計: {lm.get('loss'):.4f}" if lm.get('loss') is not None else "總計: -"
        except Exception:
            pol = val = ent = tot = "-"

        # 縮小字體顯示避免重疊
        small_font = pygame.font.SysFont(None, 22)
        self.screen.blit(small_font.render(pol, True, (200, 180, 180)), (loss_rect.left + 8, txt_y))
        self.screen.blit(small_font.render(val, True, (180, 220, 180)), (loss_rect.left + 90, txt_y))
        self.screen.blit(small_font.render(ent, True, (180, 180, 220)), (loss_rect.left + 8, txt_y + 20))
        self.screen.blit(small_font.render(tot, True, (220, 220, 140)), (loss_rect.left + 90, txt_y + 20))

        # draw simple multi-series plot (below numeric summary)
        self._draw_loss_plot(loss_rect.left + 8, loss_rect.top + 68, loss_rect.width - 16, loss_rect.height - 76)

        # leaderboard - 調整位置避免重疊
        lb_top = loss_rect.bottom + 10
        lb_title = self.font.render("排行榜:", True, (220, 220, 220))
        self.screen.blit(lb_title, (self.panel.left + 20, lb_top))
        for idx, (name, score) in enumerate(self.leaderboard[:5]):
            t = self.font.render(f"{idx+1}. {name} - {score}", True, (200, 200, 200))
            self.screen.blit(t, (self.panel.left + 20, lb_top + 24 + idx * 22))

    def draw_game_over_dialog(self):
        """繪製遊戲結束對話框"""
        # 半透明遮罩
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # 對話框
        dialog_w, dialog_h = 400, 250
        dialog_x = (self.WIDTH - dialog_w) // 2
        dialog_y = (self.HEIGHT - dialog_h) // 2
        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_w, dialog_h)
        pygame.draw.rect(self.screen, (40, 40, 50), dialog_rect, border_radius=10)
        pygame.draw.rect(self.screen, (100, 100, 120), dialog_rect, 3, border_radius=10)
        
        # 標題
        title = self.large_font.render("遊戲結束", True, (255, 100, 100))
        title_x = dialog_x + (dialog_w - title.get_width()) // 2
        self.screen.blit(title, (title_x, dialog_y + 30))
        
        # 分數
        score_text = self.large_font.render(f"最終分數: {int(self.current_score)}", True, (255, 255, 255))
        score_x = dialog_x + (dialog_w - score_text.get_width()) // 2
        self.screen.blit(score_text, (score_x, dialog_y + 80))
        
        # 按鈕
        btn_continue = pygame.Rect(dialog_x + 50, dialog_y + 140, 130, 50)
        btn_menu = pygame.Rect(dialog_x + 220, dialog_y + 140, 130, 50)
        
        pygame.draw.rect(self.screen, (80, 150, 80), btn_continue, border_radius=5)
        pygame.draw.rect(self.screen, (150, 80, 80), btn_menu, border_radius=5)
        
        continue_text = self.font.render("繼續遊玩", True, (255, 255, 255))
        menu_text = self.font.render("返回選單", True, (255, 255, 255))
        
        self.screen.blit(continue_text, (btn_continue.centerx - continue_text.get_width() // 2,
                                        btn_continue.centery - continue_text.get_height() // 2))
        self.screen.blit(menu_text, (btn_menu.centerx - menu_text.get_width() // 2,
                                     btn_menu.centery - menu_text.get_height() // 2))
        
        return btn_continue, btn_menu
    
    def draw_pause_dialog(self):
        """繪製暫停對話框"""
        # 半透明遮罩
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # 對話框
        dialog_w, dialog_h = 400, 220
        dialog_x = (self.WIDTH - dialog_w) // 2
        dialog_y = (self.HEIGHT - dialog_h) // 2
        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_w, dialog_h)
        pygame.draw.rect(self.screen, (40, 40, 50), dialog_rect, border_radius=10)
        pygame.draw.rect(self.screen, (100, 100, 120), dialog_rect, 3, border_radius=10)
        
        # 標題
        title = self.large_font.render("遊戲暫停", True, (255, 255, 100))
        title_x = dialog_x + (dialog_w - title.get_width()) // 2
        self.screen.blit(title, (title_x, dialog_y + 30))
        
        # 提示
        hint = self.font.render("按 ESC 繼續遊戲", True, (200, 200, 200))
        hint_x = dialog_x + (dialog_w - hint.get_width()) // 2
        self.screen.blit(hint, (hint_x, dialog_y + 80))
        
        # 按鈕
        btn_resume = pygame.Rect(dialog_x + 50, dialog_y + 120, 130, 50)
        btn_menu = pygame.Rect(dialog_x + 220, dialog_y + 120, 130, 50)
        
        pygame.draw.rect(self.screen, (80, 150, 80), btn_resume, border_radius=5)
        pygame.draw.rect(self.screen, (150, 80, 80), btn_menu, border_radius=5)
        
        resume_text = self.font.render("繼續遊戲", True, (255, 255, 255))
        menu_text = self.font.render("返回選單", True, (255, 255, 255))
        
        self.screen.blit(resume_text, (btn_resume.centerx - resume_text.get_width() // 2,
                                       btn_resume.centery - resume_text.get_height() // 2))
        self.screen.blit(menu_text, (btn_menu.centerx - menu_text.get_width() // 2,
                                     btn_menu.centery - menu_text.get_height() // 2))
        
        return btn_resume, btn_menu

    def handle_click(self, pos):
        # Handle game over dialog clicks
        if self.game_over:
            btn_continue, btn_menu = self.draw_game_over_dialog()  # Get button rects
            if btn_continue.collidepoint(pos):
                # 繼續遊玩 - 重置遊戲
                self.game_over = False
                self.current_score = 0.0
                return self.env.reset()
            elif btn_menu.collidepoint(pos):
                # 返回選單
                self.game_over = False
                self.running = False
                self.mode = "Menu"
                return self.env.reset()
            return None
        
        # Handle pause dialog clicks
        if self.paused:
            btn_resume, btn_menu = self.draw_pause_dialog()  # Get button rects
            if btn_resume.collidepoint(pos):
                # 繼續遊戲
                self.paused = False
                return None
            elif btn_menu.collidepoint(pos):
                # 返回選單
                self.paused = False
                self.running = False
                self.mode = "Menu"
                return self.env.reset()
            return None
        
        # If not running, these buttons start a run
        if not self.running and self.btn_human.collidepoint(pos):
            self.selected_mode = "Human"
            self.mode = "Human"
            self.running = True
            self.agent = None
            self.current_score = 0.0
            self.game_over = False
            # Reset environment and return the new state
            return self.env.reset()
        if not self.running and self.btn_ai.collidepoint(pos):
            self.selected_mode = "AI"
            self.mode = "AI"
            self.running = True
            self.current_score = 0.0
            self.game_over = False
            # ensure agent exists (try to instantiate fallback if missing)
            if self.agent is None:
                try:
                    self.agent = PPOAgent()
                except Exception:
                    self.agent = None
            # Reset environment and return the new state
            return self.env.reset()
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
                    # ESC 鍵暫停/取消暫停
                    if event.key == pygame.K_ESCAPE and self.running and not self.game_over:
                        self.paused = not self.paused
                    # 空白鍵跳躍（只在遊戲進行中且未暫停時）
                    elif self.running and self.mode == "Human" and event.key == pygame.K_SPACE and not self.paused and not self.game_over:
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
            
            # 如果遊戲暫停或結束，只渲染不更新
            if self.paused or self.game_over:
                self.screen.fill(self.BG_COLOR)
                self.draw_playfield(s)
                self.draw_panel()
                if self.paused:
                    self.draw_pause_dialog()
                elif self.game_over:
                    self.draw_game_over_dialog()
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

            # update current score
            try:
                self.current_score += float(r)
            except Exception:
                pass

            if done:
                # 在 Human 模式下顯示 Game Over 對話框
                if self.mode == "Human":
                    self.game_over = True
                    # 記錄到排行榜
                    name = "人類"
                    self.leaderboard.append((name, int(self.current_score)))
                    # keep top 10 entries sorted by score desc
                    self.leaderboard = sorted(self.leaderboard, key=lambda x: x[1], reverse=True)[:10]
                    # persist leaderboard
                    try:
                        self._save_scores()
                    except Exception:
                        pass
                else:
                    # AI 模式自動重新開始
                    name = "AI"
                    self.leaderboard.append((name, int(self.current_score)))
                    self.leaderboard = sorted(self.leaderboard, key=lambda x: x[1], reverse=True)[:10]
                    try:
                        self._save_scores()
                    except Exception:
                        pass
                    self.current_score = 0.0
                    self.n += 1
                    s = self.env.reset()

            # render
            self.screen.fill(self.BG_COLOR)
            self.draw_playfield(s)
            self.draw_panel()
            if self.game_over:
                self.draw_game_over_dialog()
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
