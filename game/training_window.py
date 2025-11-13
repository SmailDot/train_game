"""
獨立的訓練視覺化視窗
顯示神經網路結構和 Loss Function 圖表
"""
import pygame
import math
import threading
from typing import Optional, Dict, List
import numpy as np


class TrainingWindow:
    """AI 訓練時的獨立視覺化視窗"""
    WIDTH = 800
    HEIGHT = 600
    BG_COLOR = (15, 15, 20)
    FPS = 30
    
    def __init__(self):
        """初始化訓練視窗"""
        self.screen = None
        self.clock = None
        self.running = False
        self.thread = None
        
        # 數據存儲（線程安全）
        self._lock = threading.Lock()
        self.loss_history = {"policy": [], "value": [], "entropy": [], "total": []}
        self.latest_metrics = {}
        self.network_weights = None
        self.current_iteration = 0
        
        # 字體
        self.font = None
        self.large_font = None
        self.title_font = None
        
    def _init_pygame(self):
        """在新線程中初始化 Pygame"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("AI Training Visualization")
        self.clock = pygame.time.Clock()
        
        # 載入字體
        chinese_fonts = ['microsoftyahei', 'simhei', 'arial']
        for font_name in chinese_fonts:
            try:
                self.font = pygame.font.SysFont(font_name, 20)
                self.large_font = pygame.font.SysFont(font_name, 28)
                self.title_font = pygame.font.SysFont(font_name, 32)
                test = self.font.render("測試", True, (255, 255, 255))
                if test.get_width() > 0:
                    break
            except Exception:
                continue
        
        if self.font is None:
            self.font = pygame.font.Font(None, 20)
            self.large_font = pygame.font.Font(None, 28)
            self.title_font = pygame.font.Font(None, 32)
    
    def update_data(self, metrics: Dict):
        """更新訓練數據（從主線程調用）"""
        with self._lock:
            if metrics.get("policy_loss") is not None:
                self.loss_history["policy"].append(float(metrics["policy_loss"]))
            if metrics.get("value_loss") is not None:
                self.loss_history["value"].append(float(metrics["value_loss"]))
            if metrics.get("entropy") is not None:
                self.loss_history["entropy"].append(float(metrics["entropy"]))
            if metrics.get("loss") is not None:
                self.loss_history["total"].append(float(metrics["loss"]))
            
            self.latest_metrics = dict(metrics)
            if metrics.get("it") is not None:
                self.current_iteration = int(metrics["it"])
    
    def draw_neural_network(self):
        """繪製神經網路視覺化（科技感風格）"""
        # 網路結構定義
        nn_x = 50
        nn_y = 50
        nn_width = 700
        nn_height = 250
        
        # 繪製標題
        title = self.title_font.render(f"Neural Network (Iteration n={self.current_iteration})", True, (100, 200, 255))
        self.screen.blit(title, (nn_x, nn_y - 40))
        
        # 定義網路層結構：[輸入層, 隱藏層1, 隱藏層2, 輸出層]
        layers = [5, 64, 64, 2]  # 實際的 Actor-Critic 網路結構
        layer_names = ["Input\n(State)", "Hidden 1", "Hidden 2", "Output\n(Action)"]
        
        # 計算節點位置
        layer_spacing = nn_width / (len(layers) + 1)
        node_positions = []
        
        for i, num_nodes in enumerate(layers):
            layer_x = nn_x + (i + 1) * layer_spacing
            positions = []
            node_spacing = nn_height / (num_nodes + 1)
            
            # 如果節點太多，只顯示部分節點
            display_nodes = min(num_nodes, 8)
            for j in range(display_nodes):
                if display_nodes < num_nodes and j == display_nodes - 1:
                    # 最後一個節點代表省略
                    node_y = nn_y + nn_height / 2
                else:
                    node_y = nn_y + (j + 1) * (nn_height / (display_nodes + 1))
                positions.append((layer_x, node_y))
            node_positions.append(positions)
        
        # 繪製連接線（漸變效果）
        for i in range(len(node_positions) - 1):
            for start_pos in node_positions[i]:
                for end_pos in node_positions[i + 1]:
                    # 計算線條顏色（藍色到紫色漸變）
                    progress = i / (len(node_positions) - 1)
                    r = int(100 + progress * 100)
                    g = int(150 - progress * 100)
                    b = int(255 - progress * 50)
                    
                    # 繪製半透明連接線
                    pygame.draw.line(self.screen, (r, g, b), start_pos, end_pos, 1)
        
        # 繪製節點
        for i, positions in enumerate(node_positions):
            for j, pos in enumerate(positions):
                # 節點顏色（科技藍到科技紫）
                progress = i / (len(layers) - 1)
                r = int(80 + progress * 120)
                g = int(120 + progress * 30)
                b = int(255 - progress * 50)
                
                # 繪製節點（外圈光暈效果）
                pygame.draw.circle(self.screen, (r // 3, g // 3, b // 3), pos, 12)
                pygame.draw.circle(self.screen, (r, g, b), pos, 8)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 3)
                
                # 如果是省略符號
                if j == len(positions) - 1 and len(positions) < layers[i]:
                    text = self.font.render("...", True, (200, 200, 200))
                    self.screen.blit(text, (pos[0] - 10, pos[1] + 15))
        
        # 繪製層標籤
        for i, (name, num_nodes) in enumerate(zip(layer_names, layers)):
            layer_x = nn_x + (i + 1) * layer_spacing
            label_y = nn_y + nn_height + 20
            
            lines = name.split('\n')
            for idx, line in enumerate(lines):
                text = self.font.render(line, True, (180, 180, 200))
                text_rect = text.get_rect(center=(layer_x, label_y + idx * 20))
                self.screen.blit(text, text_rect)
            
            # 顯示節點數量
            count_text = self.font.render(f"({num_nodes})", True, (120, 120, 150))
            count_rect = count_text.get_rect(center=(layer_x, label_y + len(lines) * 20 + 10))
            self.screen.blit(count_text, count_rect)
    
    def draw_loss_function(self):
        """繪製 Loss Function 圖表"""
        loss_x = 50
        loss_y = 350
        loss_width = 700
        loss_height = 200
        
        # 繪製標題
        title = self.large_font.render("Loss Function", True, (255, 150, 100))
        self.screen.blit(title, (loss_x, loss_y - 35))
        
        # 繪製背景框
        pygame.draw.rect(self.screen, (25, 25, 35), (loss_x, loss_y, loss_width, loss_height))
        pygame.draw.rect(self.screen, (60, 60, 80), (loss_x, loss_y, loss_width, loss_height), 2)
        
        # 獲取數據
        with self._lock:
            lh_copy = {k: list(v) for k, v in self.loss_history.items()}
        
        max_len = max((len(v) for v in lh_copy.values()), default=0)
        
        if max_len < 2:
            hint = self.font.render("Waiting for training data...", True, (150, 150, 150))
            self.screen.blit(hint, (loss_x + 20, loss_y + loss_height // 2))
            return
        
        # 繪製圖例和最新值
        legend_x = loss_x + 10
        legend_y = loss_y + 10
        series_info = [
            ("Policy", "policy", (255, 100, 100)),
            ("Value", "value", (100, 255, 150)),
            ("Entropy", "entropy", (150, 150, 255)),
            ("Total", "total", (255, 255, 100))
        ]
        
        for idx, (name, key, color) in enumerate(series_info):
            y_pos = legend_y + idx * 25
            # 顏色方塊
            pygame.draw.rect(self.screen, color, (legend_x, y_pos, 15, 15))
            # 標籤和最新值
            latest_val = lh_copy[key][-1] if lh_copy.get(key) else 0.0
            text = self.font.render(f"{name}: {latest_val:.4f}", True, color)
            self.screen.blit(text, (legend_x + 20, y_pos))
        
        # 繪製圖表
        plot_x = loss_x + 150
        plot_y = loss_y + 10
        plot_width = loss_width - 160
        plot_height = loss_height - 20
        
        N = min(max_len, plot_width)
        
        for key, color in [("policy", (255, 100, 100)), ("value", (100, 255, 150)),
                          ("entropy", (150, 150, 255)), ("total", (255, 255, 100))]:
            seq = lh_copy.get(key, [])
            if not seq:
                continue
            
            seq = seq[-N:]
            if len(seq) < 2:
                continue
            
            mx = max(seq)
            mn = min(seq)
            denom = mx - mn if mx != mn else 1.0
            
            points = []
            for i, val in enumerate(seq):
                px = plot_x + int(i * plot_width / max(N - 1, 1))
                py = plot_y + plot_height - int((val - mn) / denom * (plot_height - 4))
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
    
    def _run_loop(self):
        """視窗主循環（在獨立線程中運行）"""
        self._init_pygame()
        self.running = True
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            # 清空畫面
            self.screen.fill(self.BG_COLOR)
            
            # 繪製內容
            self.draw_neural_network()
            self.draw_loss_function()
            
            # 更新顯示
            pygame.display.flip()
            self.clock.tick(self.FPS)
        
        pygame.quit()
    
    def start(self):
        """啟動訓練視窗（非阻塞）"""
        if self.thread is not None and self.thread.is_alive():
            return  # 已經在運行
        
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止訓練視窗"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
