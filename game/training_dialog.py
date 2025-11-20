"""
訓練模型選擇對話框
"""

import os
from typing import Optional, Tuple

import pygame


class TrainingDialog:
    """訓練配置對話框"""

    def __init__(self, screen_width: int, screen_height: int):
        # 對話框尺寸
        self.dialog_width = 600
        self.dialog_height = 500
        self.x = (screen_width - self.dialog_width) // 2
        self.y = (screen_height - self.dialog_height) // 2

        # 顏色
        self.bg_color = (40, 44, 52)
        self.border_color = (97, 218, 251)
        self.text_color = (229, 233, 240)
        self.button_color = (61, 89, 171)
        self.button_hover_color = (80, 120, 200)
        self.radio_selected_color = (97, 218, 251)
        self.radio_unselected_color = (100, 100, 100)

        # 字體
        self.title_font = pygame.font.Font(None, 36)
        self.label_font = pygame.font.Font(None, 28)
        self.button_font = pygame.font.Font(None, 24)

        # 算法選項
        self.algorithms = [
            ("PPO", "Proximal Policy Optimization - 穩定高效"),
        ]
        self.selected_algorithm = 0  # 預設選擇 PPO

        # Checkpoint 選項
        # 預設啟用 checkpoint 並自動選擇最佳檢查點
        self.use_checkpoint = True
        self.selected_checkpoint_path = self._get_default_checkpoint()

        # 按鈕位置
        self.button_height = 40
        self.button_y = self.y + self.dialog_height - 60

        self.start_button_rect = pygame.Rect(
            self.x + 50, self.button_y, 200, self.button_height
        )
        self.cancel_button_rect = pygame.Rect(
            self.x + self.dialog_width - 250, self.button_y, 200, self.button_height
        )
        self.browse_button_rect = pygame.Rect(
            self.x + 400, self.y + 340, 150, 30
        )  # checkpoint 瀏覽按鈕

        # Checkbox 位置
        self.checkbox_rect = pygame.Rect(self.x + 50, self.y + 340, 20, 20)

        self.result = None  # 對話框結果

    def _get_default_checkpoint(self) -> Optional[str]:
        """獲取預設的 checkpoint 路徑

        優先順序:
        1. checkpoint_best.pt (最佳檢查點)
        2. 最新的 checkpoint_*.pt
        3. None (從頭開始)
        """
        # 優先使用最佳檢查點
        best_checkpoint = os.path.join("checkpoints", "checkpoint_best.pt")
        if os.path.exists(best_checkpoint):
            return best_checkpoint

        # 如果沒有最佳檢查點，找最新的
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            return None

        checkpoints = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".pt"):
                try:
                    # 提取迭代數字
                    iter_num = int(
                        filename.replace("checkpoint_", "").replace(".pt", "")
                    )
                    checkpoints.append(
                        (iter_num, os.path.join(checkpoint_dir, filename))
                    )
                except ValueError:
                    continue

        if checkpoints:
            # 返回最新的 checkpoint
            checkpoints.sort(reverse=True)
            return checkpoints[0][1]

        return None

    def draw(self, screen: pygame.Surface):
        """繪製對話框"""
        # 半透明背景遮罩
        overlay = pygame.Surface((screen.get_width(), screen.get_height()))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))

        # 對話框背景
        dialog_rect = pygame.Rect(self.x, self.y, self.dialog_width, self.dialog_height)
        pygame.draw.rect(screen, self.bg_color, dialog_rect)
        pygame.draw.rect(screen, self.border_color, dialog_rect, 2)

        # 標題
        title = self.title_font.render("選擇訓練配置", True, self.text_color)
        title_rect = title.get_rect(
            center=(self.x + self.dialog_width // 2, self.y + 30)
        )
        screen.blit(title, title_rect)

        # 算法選擇標籤
        label = self.label_font.render("選擇演算法:", True, self.text_color)
        screen.blit(label, (self.x + 50, self.y + 70))

        # 繪製算法選項（單選按鈕）
        for i, (algo_name, algo_desc) in enumerate(self.algorithms):
            y_pos = self.y + 110 + i * 40

            # 單選按鈕圓圈
            circle_center = (self.x + 60, y_pos + 10)
            color = (
                self.radio_selected_color
                if i == self.selected_algorithm
                else self.radio_unselected_color
            )
            pygame.draw.circle(screen, color, circle_center, 8, 2)
            if i == self.selected_algorithm:
                pygame.draw.circle(screen, self.radio_selected_color, circle_center, 4)

            # 算法名稱和描述
            name_text = self.label_font.render(algo_name, True, self.text_color)
            screen.blit(name_text, (self.x + 80, y_pos))

            desc_text = self.button_font.render(algo_desc, True, (150, 150, 150))
            screen.blit(desc_text, (self.x + 180, y_pos + 3))

        # Checkpoint 選項
        pygame.draw.rect(
            screen,
            (
                self.radio_selected_color
                if self.use_checkpoint
                else self.radio_unselected_color
            ),
            self.checkbox_rect,
            2,
        )
        if self.use_checkpoint:
            # 打勾
            pygame.draw.line(
                screen,
                self.radio_selected_color,
                (self.checkbox_rect.x + 3, self.checkbox_rect.y + 10),
                (self.checkbox_rect.x + 8, self.checkbox_rect.y + 15),
                2,
            )
            pygame.draw.line(
                screen,
                self.radio_selected_color,
                (self.checkbox_rect.x + 8, self.checkbox_rect.y + 15),
                (self.checkbox_rect.x + 17, self.checkbox_rect.y + 5),
                2,
            )

        checkpoint_label = self.label_font.render(
            "從 Checkpoint 繼續訓練", True, self.text_color
        )
        screen.blit(checkpoint_label, (self.x + 80, self.y + 335))

        # 瀏覽按鈕
        if self.use_checkpoint:
            mouse_pos = pygame.mouse.get_pos()
            browse_color = (
                self.button_hover_color
                if self.browse_button_rect.collidepoint(mouse_pos)
                else self.button_color
            )
            pygame.draw.rect(screen, browse_color, self.browse_button_rect)
            browse_text = self.button_font.render("瀏覽...", True, self.text_color)
            browse_text_rect = browse_text.get_rect(
                center=self.browse_button_rect.center
            )
            screen.blit(browse_text, browse_text_rect)

            # 顯示選擇的檔案
            if self.selected_checkpoint_path:
                filename = os.path.basename(self.selected_checkpoint_path)
                # 如果是最佳檢查點，用綠色高亮顯示
                if filename == "checkpoint_best.pt":
                    file_color = (100, 255, 100)
                    display_text = f"✓ {filename} (推薦)"
                else:
                    file_color = (150, 150, 255)
                    display_text = f"選擇: {filename}"

                file_text = self.button_font.render(display_text, True, file_color)
                screen.blit(file_text, (self.x + 50, self.y + 380))
            else:
                # 如果沒有選擇，顯示將從頭開始
                no_file_text = self.button_font.render(
                    "將從頭開始訓練", True, (200, 200, 100)
                )
                screen.blit(no_file_text, (self.x + 50, self.y + 380))

        # 開始訓練按鈕
        mouse_pos = pygame.mouse.get_pos()
        start_color = (
            self.button_hover_color
            if self.start_button_rect.collidepoint(mouse_pos)
            else self.button_color
        )
        pygame.draw.rect(screen, start_color, self.start_button_rect)
        start_text = self.button_font.render("開始訓練", True, self.text_color)
        start_text_rect = start_text.get_rect(center=self.start_button_rect.center)
        screen.blit(start_text, start_text_rect)

        # 取消按鈕
        cancel_color = (
            (150, 50, 50)
            if self.cancel_button_rect.collidepoint(mouse_pos)
            else (100, 40, 40)
        )
        pygame.draw.rect(screen, cancel_color, self.cancel_button_rect)
        cancel_text = self.button_font.render("取消", True, self.text_color)
        cancel_text_rect = cancel_text.get_rect(center=self.cancel_button_rect.center)
        screen.blit(cancel_text, cancel_text_rect)

    def handle_click(self, pos: Tuple[int, int]) -> Optional[dict]:
        """處理點擊事件，返回選擇結果"""
        x, y = pos

        # 檢查算法選項
        for i in range(len(self.algorithms)):
            y_pos = self.y + 110 + i * 40
            radio_rect = pygame.Rect(self.x + 50, y_pos, 450, 30)
            if radio_rect.collidepoint(pos):
                self.selected_algorithm = i
                return None

        # 檢查 checkpoint checkbox
        checkbox_area = pygame.Rect(self.checkbox_rect.x, self.checkbox_rect.y, 300, 30)
        if checkbox_area.collidepoint(pos):
            self.use_checkpoint = not self.use_checkpoint
            return None

        # 檢查瀏覽按鈕
        if self.use_checkpoint and self.browse_button_rect.collidepoint(pos):
            self._open_file_dialog()
            return None

        # 檢查開始按鈕
        if self.start_button_rect.collidepoint(pos):
            algo_name = self.algorithms[self.selected_algorithm][0]
            return {
                "action": "start",
                "algorithm": algo_name,
                "checkpoint": (
                    self.selected_checkpoint_path if self.use_checkpoint else None
                ),
            }

        # 檢查取消按鈕
        if self.cancel_button_rect.collidepoint(pos):
            return {"action": "cancel"}

        return None

    def _open_file_dialog(self):
        """打開文件選擇對話框"""
        import tkinter as tk
        from tkinter import filedialog

        # 創建隱藏的 root window
        root = tk.Tk()
        root.withdraw()

        # 根據選擇的算法決定 checkpoint 目錄
        algo_name = self.algorithms[self.selected_algorithm][0].lower()
        if algo_name == "ppo":
            initial_dir = "checkpoints"
        else:
            initial_dir = f"checkpoints_{algo_name}"

        if not os.path.exists(initial_dir):
            initial_dir = "."

        # 打開文件選擇對話框
        file_path = filedialog.askopenfilename(
            title="選擇 Checkpoint 檔案",
            initialdir=initial_dir,
            filetypes=[("PyTorch 模型", "*.pt"), ("所有檔案", "*.*")],
        )

        if file_path:
            self.selected_checkpoint_path = file_path

        root.destroy()
