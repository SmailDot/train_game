"""
快速測試 UI 和訓練對話框
"""

import pygame

from game.environment import GameEnv
from game.ui import GameUI


def test_ui():
    """測試 UI 初始化和對話框"""
    print("初始化遊戲 UI...")
    env = GameEnv()
    ui = GameUI(env=env)

    print("✅ UI 初始化成功")
    print(f"視窗大小: {ui.width} x {ui.height}")
    print(f"遊戲區域: {ui.play_area}")
    print(f"狀態面板: {ui.panel}")

    # 簡短運行以測試繪製
    print("\n按 ESC 關閉測試視窗...")
    clock = pygame.time.Clock()
    s = env.reset()

    for i in range(30):  # 運行 30 幀 (~0.5秒)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return

        ui.screen.fill(ui.BG_COLOR)
        ui._draw_algorithm_panel()
        ui.draw_playfield(s)
        ui.draw_panel()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("✅ UI 測試完成 - 沒有錯誤！")


if __name__ == "__main__":
    test_ui()
