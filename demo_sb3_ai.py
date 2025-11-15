#!/usr/bin/env python3
"""
Game2048 SB3 AI 演示腳本

載入訓練好的SB3模型並在遊戲UI中展示AI表現。
"""

import argparse
import os
import sys
from pathlib import Path

import pygame
from stable_baselines3 import PPO

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _import_game_modules():
    from game.environment import GameEnv as _GameEnv
    from rl.game2048_env import Game2048Env as _Game2048Env

    return _GameEnv, _Game2048Env


GameEnv, Game2048Env = _import_game_modules()


class AIDemoUI:
    """AI演示UI類"""

    WIDTH = 1440
    HEIGHT = 840
    BG_COLOR = (30, 30, 40)
    FPS = 60

    def __init__(self, model_path: str):
        pygame.init()

        # 設置顯示
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("AI 玩遊戲演示 - SB3 模型")
        self.clock = pygame.time.Clock()

        # 載入模型
        print(f"載入模型: {model_path}")
        self.model = PPO.load(model_path)
        print("✅ 模型載入成功")

        # 創建遊戲環境
        self.game_env = GameEnv()
        self.ai_env = Game2048Env(render_mode=None)  # 不渲染，只用於獲取觀察

        # 遊戲狀態
        self.running = True
        self.paused = False
        self.score = 0
        self.steps = 0
        self.ai_control = True

        # 字體
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def draw_text(self, text: str, x: int, y: int, color=(255, 255, 255), font=None):
        """繪製文字"""
        if font is None:
            font = self.font
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def draw_game(self, state):
        """繪製遊戲畫面（基於UI類的draw_playfield邏輯）"""
        # 遊戲區域
        play_area = pygame.Rect(450, 50, 900, 700)
        pygame.draw.rect(self.screen, (20, 20, 30), play_area)

        # 從state提取遊戲狀態
        s_y = state[0]  # normalized y [0,1]
        y_px = int(s_y * play_area.height)
        y_px = max(10, min(y_px, play_area.height - 10))

        # 球的位置
        ball_x = play_area.left + int(play_area.width * 0.2)
        ball_y = play_area.top + y_px

        # 繪製障礙物
        obstacle_width = 40
        if hasattr(self.game_env, "get_all_obstacles"):
            for ob_x, gap_top, gap_bottom in self.game_env.get_all_obstacles():
                # 計算障礙物在螢幕上的位置
                ball_x_relative = int(play_area.width * 0.2)
                scale = (play_area.width - ball_x_relative) / self.game_env.MaxDist
                ob_x_px = play_area.left + int(ball_x_relative + ob_x * scale)

                # 只繪製可見的障礙物
                if play_area.left - obstacle_width < ob_x_px < play_area.right:
                    # 映射gap座標到實際高度
                    gap_top_px = play_area.top + int(
                        gap_top * play_area.height / self.game_env.ScreenHeight
                    )
                    gap_bottom_px = play_area.top + int(
                        gap_bottom * play_area.height / self.game_env.ScreenHeight
                    )

                    # 繪製上方障礙物
                    pygame.draw.rect(
                        self.screen,
                        (10, 120, 10),
                        (
                            ob_x_px,
                            play_area.top,
                            obstacle_width,
                            gap_top_px - play_area.top,
                        ),
                    )
                    # 繪製下方障礙物
                    pygame.draw.rect(
                        self.screen,
                        (10, 120, 10),
                        (
                            ob_x_px,
                            gap_bottom_px,
                            obstacle_width,
                            play_area.bottom - gap_bottom_px,
                        ),
                    )

        # 繪製球
        pygame.draw.circle(self.screen, (255, 200, 50), (ball_x, ball_y), 12)

    def reset_game(self):
        """重置遊戲"""
        obs, info = self.ai_env.reset()
        self.game_env.reset()
        self.score = 0
        self.steps = 0
        return obs

    def draw_info_panel(self, obs):
        """繪製資訊面板"""
        # 背景
        panel_rect = pygame.Rect(10, 10, 400, 250)
        pygame.draw.rect(self.screen, (50, 50, 60), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (100, 100, 120), panel_rect, 2, border_radius=10)

        # 標題
        self.draw_text("🤖 AI 狀態", 20, 20, (255, 255, 100))

        # 資訊
        y_offset = 60
        self.draw_text(f"分數: {self.score}", 20, y_offset, (255, 255, 255))
        y_offset += 30
        self.draw_text(f"步數: {self.steps}", 20, y_offset, (255, 255, 255))
        y_offset += 30
        self.draw_text(
            f"AI控制: {'開啟' if self.ai_control else '關閉'}",
            20,
            y_offset,
            (0, 255, 0) if self.ai_control else (255, 0, 0),
        )
        y_offset += 30
        self.draw_text(
            f"狀態: {'暫停' if self.paused else '運行中'}",
            20,
            y_offset,
            (255, 255, 0) if self.paused else (0, 255, 0),
        )

        # AI 決策資訊
        y_offset += 40
        self.draw_text("AI 決策資訊:", 20, y_offset, (200, 220, 255), self.small_font)
        y_offset += 25
        if hasattr(obs, "__len__") and len(obs) >= 5:
            self.draw_text(
                f"垂直位置: {obs[0]:.3f}",
                20,
                y_offset,
                (220, 220, 230),
                self.small_font,
            )
            y_offset += 20
            self.draw_text(
                f"垂直速度: {obs[1]:.3f}",
                20,
                y_offset,
                (220, 220, 230),
                self.small_font,
            )
            y_offset += 20
            self.draw_text(
                f"障礙物距離: {obs[2]:.3f}",
                20,
                y_offset,
                (220, 220, 230),
                self.small_font,
            )
            y_offset += 20
            self.draw_text(
                f"上方間隙: {obs[3]:.3f}",
                20,
                y_offset,
                (220, 220, 230),
                self.small_font,
            )
            y_offset += 20
            self.draw_text(
                f"下方間隙: {obs[4]:.3f}",
                20,
                y_offset,
                (220, 220, 230),
                self.small_font,
            )

        # 操作說明
        y_offset += 30
        self.draw_text("操作說明:", 20, y_offset, (200, 200, 200), self.small_font)
        y_offset += 20
        self.draw_text(
            "空格: 暫停/繼續", 20, y_offset, (200, 200, 200), self.small_font
        )
        y_offset += 15
        self.draw_text("A: 切換AI控制", 20, y_offset, (200, 200, 200), self.small_font)
        y_offset += 15
        self.draw_text("R: 重新開始", 20, y_offset, (200, 200, 200), self.small_font)
        y_offset += 15
        self.draw_text("ESC: 退出", 20, y_offset, (200, 200, 200), self.small_font)

    def run(self):
        """運行演示"""
        obs = self.reset_game()

        while self.running:
            # 處理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_a:
                        self.ai_control = not self.ai_control
                        print(f"AI控制: {'開啟' if self.ai_control else '關閉'}")
                    elif event.key == pygame.K_r:
                        obs = self.reset_game()
                        self.paused = False
                        print("遊戲重新開始")

            if not self.paused:
                if self.ai_control:
                    # AI 控制
                    action, _ = self.model.predict(obs, deterministic=True)

                    # 執行動作
                    obs, reward, terminated, truncated, info = self.ai_env.step(action)

                    # 更新遊戲環境 (同步動作)
                    self.game_env.step(action)

                    # 更新統計
                    self.score += reward
                    self.steps += 1

                    # 檢查遊戲結束
                    if terminated or truncated:
                        print(f"遊戲結束! 分數: {self.score}, 步數: {self.steps}")
                        obs = self.reset_game()

                else:
                    # 手動控制 (可選)
                    keys = pygame.key.get_pressed()
                    action = 0  # 預設不動
                    if keys[pygame.K_UP]:
                        action = 1  # 向上
                    elif keys[pygame.K_DOWN]:
                        action = 2  # 向下

                    if action != 0:
                        obs, reward, terminated, truncated, info = self.ai_env.step(
                            action
                        )
                        self.game_env.step(action)
                        self.score += reward
                        self.steps += 1

                        if terminated or truncated:
                            print(f"遊戲結束! 分數: {self.score}, 步數: {self.steps}")
                            obs = self.reset_game()

            # 清空螢幕
            self.screen.fill(self.BG_COLOR)

            # 繪製遊戲
            self.draw_game(obs)

            # 繪製資訊面板
            self.draw_info_panel(obs)

            # 更新顯示
            pygame.display.flip()
            self.clock.tick(self.FPS)

        pygame.quit()


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="Game2048 SB3 AI 演示")
    parser.add_argument("--model", type=str, required=True, help="SB3模型路徑")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")

    args = parser.parse_args()

    print("🎮 Game2048 SB3 AI 演示")
    print("=" * 40)
    print(f"模型: {args.model}")

    if not os.path.exists(args.model):
        print(f"❌ 模型不存在: {args.model}")
        return

    try:
        demo = AIDemoUI(args.model)
        demo.run()
    except KeyboardInterrupt:
        print("\n👋 演示結束")
    except Exception as e:
        print(f"❌ 錯誤: {e}")


if __name__ == "__main__":
    main()
