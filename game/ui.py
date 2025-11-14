import json
import math
import os
import sys
import threading
from datetime import datetime
from typing import Dict, Optional

import pygame

from agents.ppo_agent import PPOAgent
from agents.pytorch_trainer import PPOTrainer
from game.ai_manager import AlgorithmDescriptor, AlgorithmManager, AlgorithmState
from game.environment import GameEnv
from game.training_dialog import TrainingDialog

try:  # Optional advanced trainers
    from agents.q_learning_trainer import QLearningTrainer
except Exception:
    QLearningTrainer = None

try:
    from agents.sac_trainer import SACTrainer
except Exception:
    SACTrainer = None

try:
    from agents.td3_trainer import TD3Trainer
except Exception:
    TD3Trainer = None


class GameUI:
    WIDTH = 1440
    HEIGHT = 840
    BG_COLOR = (30, 30, 40)
    FPS = 60

    def __init__(self, env: Optional[GameEnv] = None, agent: Optional[PPOAgent] = None):
        pygame.init()
        self.min_width = 1200
        self.min_height = 720
        self._display_flags = pygame.RESIZABLE | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT), self._display_flags
        )
        pygame.display.set_caption("Train Game")
        self.width, self.height = self.screen.get_size()
        self.clock = pygame.time.Clock()

        # environment and agent
        self.env = env or GameEnv()
        self.agent = agent

        # start in menu mode; user must choose Human or AI to start a run
        self.mode = "Menu"
        self.selected_mode = None
        self.running = False
        self._last_layout_mode = None  # 追蹤上次佈局計算時的模式

        # fonts and counters - 使用支持中文的字體
        chinese_fonts = [
            "microsoftyahei",
            "microsoftyaheimicrosoftyaheiui",
            "microsoftyaheiui",
            "simhei",
            "simsun",
            "kaiti",
            "fangsong",
            "nsimsun",
            "msgothic",
            "mspgothic",
            "notosanscjk",
            "notosanscjksc",
            "arial",
            "verdana",
        ]

        self.font = None
        self.large_font = None

        # 嘗試載入中文字體
        for font_name in chinese_fonts:
            try:
                self.font = pygame.font.SysFont(font_name, 28)
                self.large_font = pygame.font.SysFont(font_name, 36)
                # 測試是否能正確渲染中文
                test_surface = self.font.render("測試", True, (255, 255, 255))
                if test_surface.get_width() > 0:
                    break
            except Exception:
                continue

        # 如果還是沒有找到合適的字體，使用 pygame 默認字體
        if self.font is None:
            self.font = pygame.font.Font(None, 28)
            self.large_font = pygame.font.Font(None, 36)

        self.ai_manager = AlgorithmManager()
        self.algorithm_hotkeys = {}
        self.algorithm_rects = {}
        self._register_algorithms()

        # speed/parallel controls
        self._speed_options = [1, 2, 4, 8]
        self._speed_index = 0
        self.ai_speed_multiplier = self._speed_options[self._speed_index]

        self._vector_env_options = [4, 8, 12, 16]
        self._vector_env_index = 0
        self.vector_envs = self._vector_env_options[self._vector_env_index]

        # current episode score
        self.current_score = 0.0
        # latest numeric metrics reported by trainer (thread-safe)
        self.latest_metrics = {}
        # lock for thread-safe updates from trainer thread
        self._lock = threading.Lock()
        # human jump flag (avoid double-stepping in key handler)
        self.human_jump = False

        # AI 決策信息（用於顯示）
        self.last_ai_action = None
        self.last_ai_action_prob = 0.0
        self.last_ai_value = 0.0

        # UI layout placeholders (actual geometry computed in _update_layout)
        self.play_area = pygame.Rect(0, 0, 0, 0)
        self.panel = pygame.Rect(0, 0, 0, 0)
        self.btn_human = pygame.Rect(0, 0, 0, 0)
        self.btn_ai = pygame.Rect(0, 0, 0, 0)
        self.btn_board = pygame.Rect(0, 0, 0, 0)
        self.btn_init = pygame.Rect(0, 0, 0, 0)
        self.btn_speed = pygame.Rect(0, 0, 0, 0)
        self.btn_parallel = pygame.Rect(0, 0, 0, 0)
        self.btn_multi_view = pygame.Rect(0, 0, 0, 0)  # 多視窗觀看按鈕
        self.btn_clear_board = None  # 清除排行榜按鈕（只在排行榜模式顯示）
        self._btn_save_template = pygame.Rect(0, 0, 0, 0)
        self.btn_save = None
        self._update_layout(self.width, self.height)

        # Game state flags
        self.paused = False  # ESC 暫停狀態
        self.game_over = False  # 遊戲結束狀態
        self.show_pause_menu = False  # 顯示暫停選單

        # 訓練視覺化視窗（暫不啟動第二個視窗，以免阻塞主畫面）
        self.starting_ai = False
        self._ai_init_thread = None

        # 訓練對話框
        self.training_dialog = None
        self.show_training_dialog = False

        # leaderboard entries include training iteration metadata for AI scores
        self.leaderboard = [
            {"name": "AgentA", "score": 10, "iteration": None, "note": None},
            {"name": "AgentB", "score": 7, "iteration": None, "note": None},
            {"name": "Human", "score": 3, "iteration": None, "note": None},
        ]
        # try load persisted leaderboard (if present)
        try:
            self._ensure_checkpoints()
            self._load_scores()
            self._load_training_meta()
            self._refresh_training_counters()
        except Exception:
            pass
        self._sync_vector_env_index()

        # surface for small loss plot
        self.loss_surf_size = (self.panel.width - 40, 120)
        self.loss_surf = pygame.Surface(self.loss_surf_size)

    def _register_algorithms(self) -> None:
        # 檢查 GPU 可用性
        import torch

        use_cuda = torch.cuda.is_available()
        device_str = "cuda" if use_cuda else "cpu"
        if use_cuda:
            print(f"✅ 檢測到 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 版本: {torch.version.cuda}")
        else:
            print("⚠️  未檢測到 GPU，使用 CPU 訓練")

        # 根據 training_config 設定參數
        from utils.training_config import TrainingConfig

        config = TrainingConfig(use_gpu=use_cuda)
        ppo_kwargs = config.get_ppo_kwargs()

        descriptors = [
            AlgorithmDescriptor(
                key="ppo",
                name="PPO",
                trainer_factory=lambda: PPOTrainer(**ppo_kwargs),
                use_vector_envs=True,
                vector_envs=config.get_recommended_n_envs(),
                hotkey=pygame.K_1,
                action_label="1",
                color=(120, 200, 255),
                window_title=f"PPO 訓練視窗 ({device_str.upper()})",
            )
        ]

        if SACTrainer is not None:
            descriptors.append(
                AlgorithmDescriptor(
                    key="sac",
                    name="SAC",
                    trainer_factory=SACTrainer,
                    use_vector_envs=False,
                    vector_envs=1,
                    hotkey=pygame.K_2,
                    action_label="2",
                    color=(180, 255, 180),
                    window_title="SAC 訓練視窗",
                )
            )

        if TD3Trainer is not None:
            descriptors.append(
                AlgorithmDescriptor(
                    key="td3",
                    name="TD3",
                    trainer_factory=TD3Trainer,
                    use_vector_envs=False,
                    vector_envs=1,
                    hotkey=pygame.K_3,
                    action_label="3",
                    color=(255, 200, 150),
                    window_title="TD3 訓練視窗",
                )
            )

        if QLearningTrainer is not None:
            descriptors.append(
                AlgorithmDescriptor(
                    key="dqn",
                    name="DQN",
                    trainer_factory=lambda: QLearningTrainer(mode="dqn"),
                    use_vector_envs=False,
                    vector_envs=1,
                    hotkey=pygame.K_4,
                    action_label="4",
                    color=(200, 180, 255),
                    window_title="DQN 訓練視窗",
                )
            )
            descriptors.append(
                AlgorithmDescriptor(
                    key="double_dqn",
                    name="Double DQN",
                    trainer_factory=lambda: QLearningTrainer(mode="double_dqn"),
                    use_vector_envs=False,
                    vector_envs=1,
                    hotkey=pygame.K_5,
                    action_label="5",
                    color=(220, 160, 255),
                    window_title="Double DQN 訓練視窗",
                )
            )

        for desc in descriptors:
            self.ai_manager.register(desc)
            if desc.hotkey is not None:
                self.algorithm_hotkeys[desc.hotkey] = desc.key

    def _active_slot(self) -> Optional[AlgorithmState]:
        return self.ai_manager.active_state()

    def _slot_attr(self, attr: str, default=None):
        slot = self._active_slot()
        if slot is None:
            return default
        return getattr(slot, attr, default)

    def _set_slot_attr(self, attr: str, value) -> None:
        slot = self._active_slot()
        if slot is not None:
            setattr(slot, attr, value)

    @property
    def training_iterations(self) -> int:
        value = self._slot_attr("iterations", 0)
        return int(value or 0)

    @training_iterations.setter
    def training_iterations(self, value: int) -> None:
        self._set_slot_attr("iterations", int(max(0, value)))

    @property
    def ai_round(self) -> int:
        value = self._slot_attr("ai_round", 0)
        return int(value or 0)

    @ai_round.setter
    def ai_round(self, value: int) -> None:
        self._set_slot_attr("ai_round", int(max(0, value)))

    @property
    def n(self) -> int:
        value = self._slot_attr("n", 0)
        return int(value or 0)

    @n.setter
    def n(self, value: int) -> None:
        self._set_slot_attr("n", int(max(0, value)))

    @property
    def viewer_round(self) -> int:
        value = self._slot_attr("viewer_round", 0)
        return int(value or 0)

    @viewer_round.setter
    def viewer_round(self, value: int) -> None:
        self._set_slot_attr("viewer_round", int(max(0, value)))

    @property
    def ai_status(self) -> str:
        return self._slot_attr("status", "idle") or "idle"

    @ai_status.setter
    def ai_status(self, value: str) -> None:
        self._set_slot_attr("status", str(value))

    @property
    def agent_ready(self) -> bool:
        return bool(self._slot_attr("agent_ready", False))

    @agent_ready.setter
    def agent_ready(self, value: bool) -> None:
        self._set_slot_attr("agent_ready", bool(value))

    # ------------------------------------------------------------------
    # Algorithm management helpers
    # ------------------------------------------------------------------

    def _set_active_algorithm(self, key: str) -> None:
        try:
            slot = self.ai_manager.state(key)
        except KeyError:
            return

        try:
            self.ai_manager.set_active(key)
        except KeyError:
            return

        if slot is None:
            return

        self.agent = slot.agent
        self.agent_ready = bool(slot.agent_ready)
        self.ai_status = slot.status or "idle"
        self.latest_metrics = dict(slot.latest_metrics)
        self.current_score = 0.0
        self.last_ai_action = None
        self.last_ai_action_prob = 0.0
        self.last_ai_value = 0.0

    def _handle_algorithm_toggle(self, key: str, *, force_reset: bool = False) -> None:
        if key != self.ai_manager.active_key:
            self._set_active_algorithm(key)
        slot = self._active_slot()
        if slot is None or slot.descriptor.key != key:
            return

        running = slot.trainer_thread is not None and slot.trainer_thread.is_alive()
        if running:
            print(f"🛑 停止 {slot.descriptor.name} 訓練")
            self.ai_manager.stop(key, wait=True)
            if key == self.ai_manager.active_key:
                self.agent_ready = False
                self.ai_status = "idle"
        else:
            print(f"🚀 啟動 {slot.descriptor.name} 訓練")
            self._start_algorithm_training(
                key=key, force_reset=force_reset, async_mode=False
            )

    def _stop_algorithm_training(
        self, key: Optional[str] = None, wait: bool = True
    ) -> None:
        target = key or self.ai_manager.active_key
        if target is None:
            return
        self.ai_manager.stop(target, wait=wait)
        if target == self.ai_manager.active_key:
            self.agent_ready = False
            self.ai_status = "idle"

    def _start_algorithm_training(
        self,
        key: Optional[str] = None,
        *,
        force_reset: bool = False,
        async_mode: bool = False,
    ) -> None:
        target_key = key or self.ai_manager.active_key
        if target_key is None:
            print("⚠️ 尚未選擇任何演算法，無法啟動訓練。")
            return

        try:
            slot = self.ai_manager.state(target_key)
        except KeyError:
            print(f"⚠️ 找不到演算法 {target_key}")
            return

        if slot.trainer_thread is not None and slot.trainer_thread.is_alive():
            if target_key == self.ai_manager.active_key:
                self.agent = slot.agent
                self.agent_ready = slot.agent_ready
            return

        def _setup_callback(state: AlgorithmState) -> None:
            if state.descriptor.key == "ppo":
                self._prepare_ppo_resume(state, force_reset=force_reset)
            elif force_reset:
                state.iterations = 0
                state.ai_round = 0
                state.n = 0

        vector_override = None
        if slot.descriptor.use_vector_envs:
            vector_override = self.vector_envs

        def _runner():
            try:
                self.ai_manager.start(
                    key=target_key,
                    env_factory=GameEnv,
                    metrics_consumer=self._handle_metrics_from_manager,
                    force_reset=force_reset,
                    setup_callback=_setup_callback,
                    vector_env_override=vector_override,
                )
                if target_key == self.ai_manager.active_key:
                    self.agent = slot.agent
                    self.agent_ready = slot.agent_ready
                    self.ai_status = slot.status
            finally:
                if async_mode:
                    self.starting_ai = False

        if async_mode:
            threading.Thread(target=_runner, daemon=True).start()
        else:
            _runner()

    def _handle_metrics_from_manager(self, key: str, metrics: Dict[str, float]) -> None:
        if key == self.ai_manager.active_key:
            with self._lock:
                self.latest_metrics = dict(metrics)

    def _prepare_ppo_resume(
        self, slot: AlgorithmState, *, force_reset: bool = False
    ) -> None:
        trainer = slot.trainer
        agent = slot.agent
        if trainer is None or agent is None:
            return
        if not isinstance(trainer, PPOTrainer):
            return

        try:
            import torch
        except Exception:
            print("⚠️ 找不到 PyTorch，無法載入 PPO 模型。")
            return

        def _load_model(path: str) -> bool:
            try:
                print(f"🔄 正在載入檢查點: {path}")
                state = torch.load(path, map_location=trainer.device)
                if isinstance(state, dict):
                    model_state = state.get("model_state", state)

                    # 記錄載入前的權重（用於驗證）
                    first_param_before = next(
                        iter(trainer.net.parameters())
                    ).data.clone()

                    trainer.net.load_state_dict(model_state)

                    # 檢查載入後的權重是否改變
                    first_param_after = next(iter(trainer.net.parameters())).data
                    diff = (
                        torch.abs(first_param_after - first_param_before).sum().item()
                    )

                    if diff > 1e-6:
                        print(f"   ✅ 模型權重已成功載入 (權重差異: {diff:.2f})")
                    else:
                        print(f"   ⚠️  警告: 權重似乎未改變 (差異: {diff:.6f})")

                    opt_state = state.get("optimizer_state")
                    if opt_state is not None:
                        try:
                            trainer.opt.load_state_dict(opt_state)
                            print("   ✅ 優化器狀態已載入")
                        except Exception:
                            print("   ⚠️ 無法載入 optimizer_state，將重新初始化優化器")
                    return True
                else:
                    print("   ❌ 檢查點格式錯誤（不是字典）")
            except Exception as load_err:
                print(f"   ❌ 載入模型失敗: {load_err}")
            return False

        checkpoint_path = None
        if not force_reset and self.training_iterations > 0:
            candidate = os.path.join(
                "checkpoints", f"checkpoint_{self.training_iterations}.pt"
            )
            if os.path.exists(candidate):
                checkpoint_path = candidate

        if checkpoint_path is None and not force_reset:
            latest_path, latest_iter = self._latest_checkpoint()
            if latest_path is not None:
                checkpoint_path = latest_path
                if (
                    isinstance(latest_iter, int)
                    and latest_iter > self.training_iterations
                ):
                    self.training_iterations = latest_iter
                    self.n = latest_iter

        loaded = False
        if checkpoint_path is not None and not force_reset:
            self.ai_status = "loading"
            print(f"\n{'='*60}")
            print("📥 開始載入檢查點")
            print(f"{'='*60}")
            loaded = _load_model(checkpoint_path)
            if loaded:
                print("✅ 檢查點載入成功！")
            else:
                print("❌ 檢查點載入失敗")
            print(f"{'='*60}\n")

        # 嘗試載入最佳檢查點（如果存在且未載入其他檢查點）
        if not loaded and not force_reset:
            best_checkpoint = os.path.join("checkpoints", "checkpoint_best.pt")
            if os.path.exists(best_checkpoint):
                self.ai_status = "loading"
                print(f"\n💎 嘗試載入最佳檢查點: {best_checkpoint}")
                loaded = _load_model(best_checkpoint)
                if loaded:
                    print("✅ 最佳檢查點載入成功！")

        if not loaded and not force_reset:
            model_path = os.path.join("checkpoints", "ppo_best.pth")
            if os.path.exists(model_path):
                self.ai_status = "loading"
                print(f"\n🔄 嘗試載入備用檢查點: {model_path}")
                loaded = _load_model(model_path)

        if not loaded and force_reset:
            self.training_iterations = 0
            self.n = 0
            slot.ai_round = 0

        slot.agent = agent
        slot.trainer = trainer

    def _update_layout(self, width: int, height: int) -> None:
        width = int(max(self.min_width, width))
        height = int(max(self.min_height, height))
        self.width = width
        self.height = height
        self.WIDTH = width
        self.HEIGHT = height

        # 調整布局：左側為演算法控制區，右側為狀態面板
        # 當 AI 訓練時，演算法面板隱藏，遊戲區域擴展到左側
        algo_panel_width = max(380, int(width * 0.32))
        if algo_panel_width > width - 420:
            algo_panel_width = max(360, width - 420)

        status_panel_width = max(320, int(width * 0.25))

        # 根據模式動態調整遊戲區域
        if self.mode == "AI" and self.running:
            # AI 訓練時，遊戲區域從最左邊開始
            play_start_x = 0
            play_width = max(400, width - status_panel_width)
        else:
            # 其他模式，遊戲區域從演算法面板右側開始
            play_start_x = algo_panel_width
            play_width = max(400, width - algo_panel_width - status_panel_width)

        self.algo_panel = pygame.Rect(0, 0, algo_panel_width, height)
        self.play_area = pygame.Rect(play_start_x, 0, play_width, height)
        self.panel = pygame.Rect(
            play_start_x + play_width, 0, status_panel_width, height
        )

        btn_width = self.panel.width - 40
        btn_height = 56
        left = self.panel.left + 20
        top = 40
        spacing = 14

        self.btn_human = pygame.Rect(left, top, btn_width, btn_height)
        top += btn_height + spacing
        self.btn_ai = pygame.Rect(left, top, btn_width, btn_height)
        top += btn_height + spacing
        self.btn_board = pygame.Rect(left, top, btn_width, btn_height)
        top += btn_height + spacing
        self.btn_multi_view = pygame.Rect(left, top, btn_width, btn_height)
        top += btn_height + spacing
        self.btn_init = pygame.Rect(left, top, btn_width, btn_height)
        top += btn_height + spacing
        self.btn_speed = pygame.Rect(left, top, btn_width, btn_height)
        top += btn_height + spacing
        self.btn_parallel = pygame.Rect(left, top, btn_width, btn_height)
        top += btn_height + spacing

        save_bottom_margin = 30
        save_y = max(top, self.panel.bottom - btn_height - save_bottom_margin)
        self._btn_save_template = pygame.Rect(left, save_y, btn_width, btn_height)
        if self.btn_save is not None:
            self.btn_save = self._btn_save_template.copy()

        self.loss_surf_size = (btn_width, max(120, int(self.panel.height * 0.18)))
        self.loss_surf = pygame.Surface(self.loss_surf_size)

    def _handle_resize(self, width: int, height: int) -> None:
        width = int(max(self.min_width, width))
        height = int(max(self.min_height, height))
        self.screen = pygame.display.set_mode((width, height), self._display_flags)
        pygame.display.set_caption(f"Train Game ({width}x{height})")
        self._update_layout(width, height)

    def _sync_vector_env_index(self) -> None:
        options = set(int(v) for v in self._vector_env_options)
        options.add(int(max(1, self.vector_envs)))
        self._vector_env_options = sorted(options)
        try:
            self._vector_env_index = self._vector_env_options.index(
                int(self.vector_envs)
            )
        except ValueError:
            self._vector_env_index = 0

    def _handle_toggle_speed(self) -> None:
        self._speed_index = (self._speed_index + 1) % len(self._speed_options)
        self.ai_speed_multiplier = self._speed_options[self._speed_index]
        print(f"觀戰速度切換為 x{self.ai_speed_multiplier}")

    def _handle_cycle_parallel_envs(self) -> None:
        if not self._vector_env_options:
            self._vector_env_options = [4]
        self._vector_env_index = (self._vector_env_index + 1) % len(
            self._vector_env_options
        )
        self.vector_envs = max(1, int(self._vector_env_options[self._vector_env_index]))
        self._sync_vector_env_index()
        print(f"訓練並行環境數設定為 {self.vector_envs}")
        try:
            slot = self._active_slot()
            if slot is not None:
                self._save_training_meta(slot.iterations, slot.ai_round)
        except Exception:
            pass
        slot = self._active_slot()
        if (
            slot is not None
            and slot.trainer_thread is not None
            and slot.trainer_thread.is_alive()
        ):
            print("⚠️ 目前訓練仍在進行，新設定將於下次初始化生效。")

    def draw_playfield(self, state):
        # Render leaderboard view when排行榜模式
        if self.mode == "Board":
            self._draw_leaderboard_view()
            return

        # draw background for play area
        pygame.draw.rect(self.screen, (20, 20, 30), self.play_area)

        # map env coords to pixels
        # state[0] is normalized y (0..1), map to play_area height
        s_y = state[0]  # normalized y [0,1]
        # 使用 play_area 的實際高度而不是 env.ScreenHeight
        y_px = int(s_y * self.play_area.height)
        # clamp to visible range within play_area
        y_px = max(10, min(y_px, self.play_area.height - 10))

        # ball: fixed x position at 20% of play area width (相對於 play_area)
        ball_x = self.play_area.left + int(self.play_area.width * 0.2)
        ball_y = self.play_area.top + y_px

        # Draw all obstacles (scrolling from right to left)
        obstacle_width = 40
        if hasattr(self.env, "get_all_obstacles"):
            for ob_x, gap_top, gap_bottom in self.env.get_all_obstacles():
                # Map env x coordinates onto pixels.
                # env: x=0 sits at the player while x=MaxDist is the far right edge.
                # 計算相對於 play_area 左邊界的 x 位置
                ball_x_relative = int(self.play_area.width * 0.2)
                scale = (self.play_area.width - ball_x_relative) / self.env.MaxDist
                ob_x_px = self.play_area.left + int(ball_x_relative + ob_x * scale)

                # Only draw obstacles that are visible on screen
                if (
                    self.play_area.left - obstacle_width
                    < ob_x_px
                    < self.play_area.right
                ):
                    # 將 gap 座標映射到 play_area 的實際高度
                    gap_top_px = self.play_area.top + int(
                        gap_top * self.play_area.height / self.env.ScreenHeight
                    )
                    gap_bottom_px = self.play_area.top + int(
                        gap_bottom * self.play_area.height / self.env.ScreenHeight
                    )

                    # draw top pillar and bottom pillar with gap in between
                    pygame.draw.rect(
                        self.screen,
                        (10, 120, 10),
                        (
                            ob_x_px,
                            self.play_area.top,
                            obstacle_width,
                            gap_top_px - self.play_area.top,
                        ),
                    )
                    pygame.draw.rect(
                        self.screen,
                        (10, 120, 10),
                        (
                            ob_x_px,
                            gap_bottom_px,
                            obstacle_width,
                            self.play_area.bottom - gap_bottom_px,
                        ),
                    )

        # Draw ball on top of obstacles
        pygame.draw.circle(self.screen, (255, 200, 50), (ball_x, ball_y), 12)

    def _draw_leaderboard_view(self):
        pygame.draw.rect(self.screen, (20, 20, 30), self.play_area)

        title = self.large_font.render("歷史排行榜 Top 20", True, (240, 240, 240))
        subtitle = self.font.render("含 AI 訓練迭代標記", True, (180, 180, 200))

        title_x = self.play_area.left + (self.play_area.width - title.get_width()) // 2
        subtitle_x = (
            self.play_area.left + (self.play_area.width - subtitle.get_width()) // 2
        )
        self.screen.blit(title, (title_x, 40))
        self.screen.blit(subtitle, (subtitle_x, 90))

        entries = sorted(
            self.leaderboard, key=lambda x: x.get("score", 0), reverse=True
        )[:20]
        if not entries:
            empty = self.font.render("目前沒有紀錄", True, (200, 200, 200))
            empty_x = (
                self.play_area.left + (self.play_area.width - empty.get_width()) // 2
            )
            self.screen.blit(empty, (empty_x, 160))
        else:
            columns = 2
            rows_per_column = 10
            column_width = self.play_area.width // columns
            base_x = self.play_area.left + 60
            base_y = 130
            line_height = 32

            label_color = (220, 220, 230)
            value_color = (200, 200, 210)

            for idx, entry in enumerate(entries):
                column = idx // rows_per_column
                row = idx % rows_per_column
                x = base_x + column * column_width
                y = base_y + row * line_height
                name = entry.get("name", "-")
                score = int(entry.get("score", 0))
                note = entry.get("note")
                iteration = entry.get("iteration")

                rank_text = f"{idx + 1:>2}. {name:<8}"
                rank_surface = self.font.render(rank_text, True, label_color)
                self.screen.blit(rank_surface, (x, y))

                detail = f"{score:>4} 分"
                if note:
                    detail += f" {note}"
                elif entry.get("name") == "AI" and isinstance(iteration, int):
                    detail += f" (第{iteration:,}次訓練)"
                detail_surface = self.font.render(detail, True, value_color)
                self.screen.blit(detail_surface, (x + 180, y))

        # 清除歷史紀錄按鈕
        btn_width = 200
        btn_height = 50
        btn_x = self.play_area.left + (self.play_area.width - btn_width) // 2
        btn_y = self.play_area.bottom - 120
        self.btn_clear_board = pygame.Rect(btn_x, btn_y, btn_width, btn_height)

        pygame.draw.rect(
            self.screen, (180, 60, 60), self.btn_clear_board, border_radius=8
        )
        clear_text = self.large_font.render("移除歷史紀錄", True, (255, 255, 255))
        clear_x = self.btn_clear_board.centerx - clear_text.get_width() // 2
        clear_y = self.btn_clear_board.centery - clear_text.get_height() // 2
        self.screen.blit(clear_text, (clear_x, clear_y))

        hint = self.font.render("再次點擊『排行榜』返回選單", True, (150, 150, 160))
        hint_x = self.play_area.left + (self.play_area.width - hint.get_width()) // 2
        self.screen.blit(hint, (hint_x, self.play_area.bottom - 60))

    def _draw_algorithm_panel(self, top: int = None) -> int:
        """繪製演算法控制面板在左側區域"""
        keys = self.ai_manager.keys()
        if not keys:
            return 0

        # 繪製左側演算法面板背景
        pygame.draw.rect(self.screen, (25, 25, 35), self.algo_panel)

        x = self.algo_panel.left + 20
        width = self.algo_panel.width - 40
        y = 30

        title = self.large_font.render("演算法控制面板", True, (200, 220, 255))
        self.screen.blit(title, (x, y))
        y += title.get_height() + 8

        subtitle = self.font.render("按數字鍵 1-5 切換演算法", True, (160, 180, 200))
        self.screen.blit(subtitle, (x, y))
        y += subtitle.get_height() + 20

        entry_height = 85  # 增加高度避免文字重疊
        spacing = 14
        self.algorithm_rects = {}

        for idx, key in enumerate(keys):
            desc = self.ai_manager.descriptor(key)
            slot = self.ai_manager.state(key)
            if slot is None:
                continue

            rect = pygame.Rect(x, y, width, entry_height)
            active = key == self.ai_manager.active_key
            running = slot.trainer_thread is not None and slot.trainer_thread.is_alive()

            # 背景顏色（確保值在 0-255 範圍內）
            base_color = desc.color
            bg = tuple(
                min(255, max(0, int(c * (1.2 if active else 0.8)))) for c in base_color
            )
            pygame.draw.rect(self.screen, bg, rect, border_radius=10)

            # 外框（繪製在背景之上，不能同時使用 width 和 border_radius）
            outline_color = (240, 240, 255) if active else (100, 110, 130)
            outline_width = 3 if active else 2
            # 使用多次繪製來模擬圓角外框
            for i in range(outline_width):
                inflated = rect.inflate(-i * 2, -i * 2)
                pygame.draw.rect(
                    self.screen, outline_color, inflated, 1, border_radius=10
                )

            # 標籤（演算法名稱 + 快捷鍵）
            label = (
                f"[{desc.action_label}] {desc.name}" if desc.action_label else desc.name
            )
            label_surface = self.large_font.render(label, True, (20, 20, 30))
            self.screen.blit(label_surface, (rect.x + 16, rect.y + 10))

            # 狀態資訊
            status_map = {
                "initializing": "初始化中",
                "loading": "載入模型",
                "training": "訓練中",
                "saving": "儲存中",
                "saved": "已儲存",
                "resetting": "重新設定",
                "error": "錯誤",
                "idle": "待命",
            }
            status_text = status_map.get(slot.status, slot.status)
            status_color = (30, 30, 40)
            status_surface = self.font.render(
                f"狀態: {status_text}", True, status_color
            )
            self.screen.blit(status_surface, (rect.x + 16, rect.y + 40))

            # 迭代次數
            metric_text = f"迭代: {slot.iterations:,} 次"
            metric_surface = self.font.render(metric_text, True, status_color)
            self.screen.blit(metric_surface, (rect.x + 16, rect.y + 64))

            # 初始化按鈕（在左側）
            init_rect = pygame.Rect(rect.right - 180, rect.y + 12, 76, 30)
            init_color = (180, 80, 80)
            init_label = "初始化"
            pygame.draw.rect(self.screen, init_color, init_rect, border_radius=6)
            # 按鈕外框
            for i in range(2):
                inflated_btn = init_rect.inflate(-i * 2, -i * 2)
                pygame.draw.rect(
                    self.screen, (40, 40, 50), inflated_btn, 1, border_radius=6
                )

            init_text = self.font.render(init_label, True, (255, 255, 255))
            self.screen.blit(
                init_text,
                (
                    init_rect.centerx - init_text.get_width() // 2,
                    init_rect.centery - init_text.get_height() // 2,
                ),
            )

            # 啟動/停止按鈕（在右側）
            toggle_rect = pygame.Rect(rect.right - 90, rect.y + 12, 76, 62)
            toggle_color = (230, 100, 100) if running else (100, 220, 120)
            toggle_label = "停止" if running else "啟動"
            pygame.draw.rect(self.screen, toggle_color, toggle_rect, border_radius=8)
            # 按鈕外框
            for i in range(2):
                inflated_btn = toggle_rect.inflate(-i * 2, -i * 2)
                pygame.draw.rect(
                    self.screen, (40, 40, 50), inflated_btn, 1, border_radius=8
                )

            btn_text = self.large_font.render(toggle_label, True, (255, 255, 255))
            self.screen.blit(
                btn_text,
                (
                    toggle_rect.centerx - btn_text.get_width() // 2,
                    toggle_rect.centery - btn_text.get_height() // 2,
                ),
            )

            self.algorithm_rects[key] = {
                "select": rect,
                "toggle": toggle_rect,
                "init": init_rect,
            }

            y += entry_height + spacing

        return y

    def draw_panel(self):
        """繪製右側狀態面板（不包含演算法控制）"""
        if self.mode == "AI" and self.running:
            if self.btn_save is None:
                self.btn_save = self._btn_save_template.copy()
        else:
            self.btn_save = None

        pygame.draw.rect(self.screen, (18, 18, 22), self.panel)

        # buttons - 按鈕文字置中
        button_specs = [
            (self.btn_human, "人類遊玩", self.large_font, (70, 70, 80)),
            (self.btn_ai, "AI 訓練", self.large_font, (70, 70, 80)),
            (self.btn_board, "排行榜", self.large_font, (70, 70, 80)),
            (self.btn_multi_view, "多視窗觀看", self.font, (80, 70, 120)),
            (self.btn_init, "初始化訓練", self.font, (70, 70, 80)),
            (
                self.btn_speed,
                f"觀戰速度 x{self.ai_speed_multiplier}",
                self.font,
                (65, 80, 90),
            ),
            (
                self.btn_parallel,
                f"並行環境 {self.vector_envs} 個",
                self.font,
                (65, 80, 90),
            ),
        ]

        for rect, label, font_obj, color in button_specs:
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            text_surface = font_obj.render(label, True, (240, 240, 240))
            self.screen.blit(
                text_surface,
                (
                    rect.centerx - text_surface.get_width() // 2,
                    rect.centery - text_surface.get_height() // 2,
                ),
            )

        hint_surface = self.font.render(
            "提示: 點擊切換觀戰速度與並行環境", True, (160, 160, 175)
        )
        hint_y = self.btn_parallel.bottom + 10
        if self.btn_save is not None:
            reserve_y = self._btn_save_template.top - hint_surface.get_height() - 6
            hint_y = min(hint_y, max(self.btn_parallel.bottom + 4, reserve_y))
        self.screen.blit(hint_surface, (self.panel.left + 24, hint_y))

        if self.btn_save is not None:
            pygame.draw.rect(self.screen, (90, 90, 100), self.btn_save, border_radius=8)
            save_surface = self.font.render("儲存訓練", True, (235, 235, 235))
            self.screen.blit(
                save_surface,
                (
                    self.btn_save.centerx - save_surface.get_width() // 2,
                    self.btn_save.centery - save_surface.get_height() // 2,
                ),
            )

        # mode indicator & current score - 使用更大的間距
        info_y = hint_y + hint_surface.get_height() + 20
        mode_map = {"Human": "人類", "AI": "AI 訓練", "Menu": "選單", "Board": "排行榜"}
        mode_name = mode_map.get(self.mode, str(self.mode))
        mode_text = self.large_font.render("模式", True, (150, 150, 160))
        mode_value = self.large_font.render(mode_name, True, (220, 220, 230))
        self.screen.blit(mode_text, (self.panel.left + 20, info_y))
        self.screen.blit(mode_value, (self.panel.left + 20, info_y + 34))

        # current score - 更醒目的分數顯示
        score_y = info_y + 96
        score_label = self.large_font.render("本局分數", True, (150, 150, 160))
        score_value = self.large_font.render(
            f"{int(self.current_score)}", True, (100, 255, 100)
        )
        self.screen.blit(score_label, (self.panel.left + 20, score_y))
        self.screen.blit(score_value, (self.panel.left + 20, score_y + 35))

        ai_info_bottom = score_y + 90

        # AI 決策信息（僅在 AI 模式下顯示）
        if self.mode == "AI" and self.running:
            ai_info_y = score_y + 92
            ai_title = self.font.render("AI 訓練狀態", True, (150, 200, 255))
            self.screen.blit(ai_title, (self.panel.left + 20, ai_info_y))

            status_map = {
                "initializing": ("初始化中...", (255, 200, 120)),
                "loading": ("載入模型...", (200, 220, 255)),
                "training": ("訓練中", (120, 255, 160)),
                "saving": ("儲存中...", (255, 220, 140)),
                "saved": ("已儲存", (180, 220, 255)),
                "resetting": ("重新初始化", (255, 200, 140)),
                "error": ("發生錯誤", (255, 120, 120)),
                "idle": ("待機", (180, 180, 180)),
            }
            status_text, status_color = status_map.get(
                self.ai_status, (self.ai_status, (200, 200, 200))
            )
            status_surface = self.font.render(
                f"狀態: {status_text}", True, status_color
            )
            self.screen.blit(status_surface, (self.panel.left + 20, ai_info_y + 30))

            # 顯示目前選中的演算法名稱
            slot = self._active_slot()
            if slot is not None:
                algo_label = self.font.render(
                    f"演算法: {slot.descriptor.name}", True, (200, 200, 230)
                )
                self.screen.blit(algo_label, (self.panel.left + 20, ai_info_y + 52))
                # info_y_cursor initialized below; don't increment before it's defined

            line_height = 32  # 增加行高以避免重疊
            info_y_cursor = ai_info_y + 65
            with self._lock:
                metrics_snapshot = dict(self.latest_metrics)

            metrics_spec = [
                ("PPO 更新次數", "it", "{:,.0f}", (210, 220, 255), ""),
                ("累積訓練回合", "episode_count", "{:,.0f}", (200, 255, 200), ""),
                ("最近平均回報", "mean_reward", "{:.2f}", (255, 240, 180), ""),
                (
                    "Policy Loss",
                    "policy_loss",
                    "{:.4f}",
                    (200, 80, 80),
                    " (越低越好)",
                ),
                ("Value Loss", "value_loss", "{:.4f}", (80, 200, 120), " (越低越好)"),
                (
                    "Entropy",
                    "entropy",
                    "{:.4f}",
                    (120, 120, 200),
                    " (初期高後期低)",
                ),
            ]

            for label, key, fmt, color, note in metrics_spec:
                value = metrics_snapshot.get(key)
                if isinstance(value, (int, float)):
                    text_str = f"{label}: {fmt.format(value)}{note}"
                    text = self.font.render(text_str, True, color)
                    self.screen.blit(text, (self.panel.left + 20, info_y_cursor))
                    info_y_cursor += line_height

            if not self.agent_ready:
                waiting = self.font.render(
                    "AI 正在載入/訓練，請稍候...", True, (200, 200, 200)
                )
                self.screen.blit(waiting, (self.panel.left + 20, info_y_cursor))
                info_y_cursor += line_height
            elif self.last_ai_action is not None:
                action_text = "跳躍 🔥" if self.last_ai_action == 1 else "不動 ▪"
                action_color = (
                    (255, 200, 50) if self.last_ai_action == 1 else (150, 150, 150)
                )
                action = self.font.render(f"動作: {action_text}", True, action_color)
                self.screen.blit(action, (self.panel.left + 20, info_y_cursor))
                info_y_cursor += line_height

                # 顯示動作機率
                prob_text = f"信心: {self.last_ai_action_prob:.1%}"
                prob = self.font.render(prob_text, True, (200, 200, 200))
                self.screen.blit(prob, (self.panel.left + 20, info_y_cursor))
                info_y_cursor += line_height

                # 顯示狀態價值估計
                value_text = f"價值: {self.last_ai_value:.2f}"
                value_color = (
                    (100, 255, 100) if self.last_ai_value > 0 else (255, 100, 100)
                )
                value = self.font.render(value_text, True, value_color)
                self.screen.blit(value, (self.panel.left + 20, info_y_cursor))
                info_y_cursor += line_height

            ai_info_bottom = info_y_cursor

        # leaderboard - 簡潔顯示（只在非 AI 訓練模式下顯示）
        lb_bottom = score_y + 120  # 預設位置
        if not (self.mode == "AI" and self.running):
            lb_top = score_y + 120
            lb_title = self.large_font.render("排行榜 Top 5", True, (200, 200, 220))
            self.screen.blit(lb_title, (self.panel.left + 20, lb_top))

            sorted_entries = sorted(
                self.leaderboard, key=lambda x: x.get("score", 0), reverse=True
            )
            lb_entries = min(len(sorted_entries), 5)
            for idx, entry in enumerate(sorted_entries[:5]):
                name = entry.get("name", "-")
                score = int(entry.get("score", 0))
                iteration = entry.get("iteration")
                rank_text = f"{idx+1}. {name}: {score}"
                note = entry.get("note")
                if note:
                    rank_text += f" {note}"
                elif (
                    entry.get("name") == "AI"
                    and isinstance(iteration, int)
                    and iteration >= 0
                ):
                    rank_text += f" (第{iteration:,}次訓練)"
                t = self.font.render(rank_text, True, (180, 180, 200))
                self.screen.blit(t, (self.panel.left + 25, lb_top + 40 + idx * 28))

            lb_bottom = lb_top + 40 + lb_entries * 28
        elif self.mode == "AI" and self.running:
            # AI 訓練模式下，lb_bottom 使用 AI 信息的底部位置
            lb_bottom = (
                ai_info_bottom if "ai_info_bottom" in locals() else score_y + 120
            )

        plot_w, plot_h = self.loss_surf_size
        plot_x = self.panel.left + 20
        content_anchor = max(
            ai_info_bottom if "ai_info_bottom" in locals() else score_y + 120,
            lb_bottom + 20,
        )
        max_y = self.panel.bottom - plot_h - 20
        if max_y > self.panel.top + 40 and max_y >= content_anchor:
            plot_y = max(content_anchor, self.panel.top + 40)
            self._draw_loss_plot(plot_x, int(plot_y), int(plot_w), int(plot_h))

        # 如果在選單模式，在遊玩區域顯示提示
        if not self.running:
            title = self.large_font.render("訓練遊戲", True, (240, 240, 240))
            hint1 = self.font.render("「人類遊玩」: 你來操控", True, (200, 200, 200))
            hint2 = self.font.render("「AI 訓練」: 觀看 AI 學習", True, (200, 200, 200))
            # Center in play area
            title_x = self.play_area.left + (
                self.play_area.width // 2 - title.get_width() // 2
            )
            hint1_x = self.play_area.left + (
                self.play_area.width // 2 - hint1.get_width() // 2
            )
            hint2_x = self.play_area.left + (
                self.play_area.width // 2 - hint2.get_width() // 2
            )
            self.screen.blit(title, (title_x, 100))
            self.screen.blit(hint1, (hint1_x, 150))
            self.screen.blit(hint2, (hint2_x, 185))

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
        score_text = self.large_font.render(
            f"最終分數: {int(self.current_score)}", True, (255, 255, 255)
        )
        score_x = dialog_x + (dialog_w - score_text.get_width()) // 2
        self.screen.blit(score_text, (score_x, dialog_y + 80))

        # 按鈕
        btn_continue = pygame.Rect(dialog_x + 50, dialog_y + 140, 130, 50)
        btn_menu = pygame.Rect(dialog_x + 220, dialog_y + 140, 130, 50)

        pygame.draw.rect(self.screen, (80, 150, 80), btn_continue, border_radius=5)
        pygame.draw.rect(self.screen, (150, 80, 80), btn_menu, border_radius=5)

        continue_text = self.font.render("繼續遊玩", True, (255, 255, 255))
        menu_text = self.font.render("返回選單", True, (255, 255, 255))

        self.screen.blit(
            continue_text,
            (
                btn_continue.centerx - continue_text.get_width() // 2,
                btn_continue.centery - continue_text.get_height() // 2,
            ),
        )
        self.screen.blit(
            menu_text,
            (
                btn_menu.centerx - menu_text.get_width() // 2,
                btn_menu.centery - menu_text.get_height() // 2,
            ),
        )

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

        self.screen.blit(
            resume_text,
            (
                btn_resume.centerx - resume_text.get_width() // 2,
                btn_resume.centery - resume_text.get_height() // 2,
            ),
        )
        self.screen.blit(
            menu_text,
            (
                btn_menu.centerx - menu_text.get_width() // 2,
                btn_menu.centery - menu_text.get_height() // 2,
            ),
        )

        return btn_resume, btn_menu

    def _start_training_with_config(self, config: dict):
        """根據對話框配置啟動訓練"""
        algorithm = config["algorithm"].lower()
        checkpoint = config.get("checkpoint")

        # 設置活躍算法
        self._set_active_algorithm(algorithm)

        # 如果有 checkpoint，載入它
        if checkpoint:
            print(f"從 checkpoint 載入：{checkpoint}")
            # TODO: 實現 checkpoint 載入邏輯
            # 需要在 trainer 中添加 load_checkpoint 方法

        # 啟動訓練
        self.selected_mode = "AI"
        self.mode = "AI"
        self.running = True
        self.current_score = 0.0
        self.game_over = False
        self.paused = False
        self.viewer_round = 0

        # 重新計算佈局以擴展遊戲區域
        self._update_layout(self.width, self.height)

        # 重置 AI 顯示資訊
        self.last_ai_action = None
        self.last_ai_action_prob = 0.0
        self.last_ai_value = 0.0
        self.agent_ready = False
        self.ai_status = "initializing"

        print(f"啟動 {algorithm.upper()} 訓練模式（背景初始化）...")
        self._start_ai_training_async(force_reset=False)

    def _start_ai_training_async(self, force_reset: bool = False):
        """在背景線程初始化並啟動 AI 訓練，避免卡住主畫面。"""
        if self.starting_ai:
            print("AI 訓練初始化中，請稍候...")
            return

        self.starting_ai = True
        self.ai_status = "initializing"

        def _worker():
            try:
                self._start_algorithm_training(
                    force_reset=force_reset, async_mode=False
                )
            except Exception as exc:
                print(f"AI 訓練初始化失敗：{exc}")
                import traceback

                traceback.print_exc()
                self.agent = None
                self.agent_ready = False
                self.ai_status = "error"
            finally:
                self.starting_ai = False

        self._ai_init_thread = threading.Thread(target=_worker, daemon=True)
        self._ai_init_thread.start()

    def _launch_multi_window_view(self):
        """啟動多視窗觀看模式"""
        import subprocess
        import sys

        print("🚀 正在啟動多視窗觀看模式...")
        print("將開啟 5 個訓練視窗（PPO, SAC, DQN, Double DQN, TD3）")

        try:
            # 使用 subprocess 在背景執行 run_multi_train.py
            script_path = "run_multi_train.py"
            subprocess.Popen(
                [sys.executable, script_path],
                creationflags=(
                    subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
                ),
            )
            print("✅ 多視窗模式已啟動！")
            print("提示：關閉所有新視窗以結束多視窗模式")
        except Exception as e:
            print(f"❌ 啟動多視窗模式失敗：{e}")
            import traceback

            traceback.print_exc()

    def handle_click(self, pos):
        # Handle training dialog clicks
        if self.show_training_dialog and self.training_dialog is not None:
            result = self.training_dialog.handle_click(pos)
            if result is not None:
                self.show_training_dialog = False
                if result["action"] == "start":
                    # 開始訓練
                    self._start_training_with_config(result)
                    return self.env.reset()
                elif result["action"] == "cancel":
                    # 取消
                    self.training_dialog = None
            return None

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
                self.agent = None
                self.agent_ready = False
                self.ai_status = "idle"
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
                self.agent = None
                self.agent_ready = False
                self.ai_status = "idle"
                return self.env.reset()
            return None

        # Save training progress (AI mode only)
        if self.btn_save is not None and self.btn_save.collidepoint(pos):
            return self._handle_save_training()

        # Initialize training reset
        if self.btn_init.collidepoint(pos):
            return self._handle_init_training()

        # Speed / parallel configuration buttons
        if self.btn_speed.collidepoint(pos):
            self._handle_toggle_speed()
            return None
        if self.btn_parallel.collidepoint(pos):
            self._handle_cycle_parallel_envs()
            return None

        # Algorithm selection / toggle / init
        for key, rects in self.algorithm_rects.items():
            toggle_rect = rects.get("toggle")
            select_rect = rects.get("select")
            init_rect = rects.get("init")

            # 初始化按鈕（優先處理，因為它在內部）
            if init_rect is not None and init_rect.collidepoint(pos):
                self._handle_init_training(algorithm_key=key)
                return None

            # 啟動/停止按鈕
            if toggle_rect is not None and toggle_rect.collidepoint(pos):
                self._handle_algorithm_toggle(key)
                return None

            # 選擇演算法
            if select_rect is not None and select_rect.collidepoint(pos):
                self._set_active_algorithm(key)
                return None

        # If not running, these buttons start a run
        if not self.running and self.btn_human.collidepoint(pos):
            self.selected_mode = "Human"
            self.mode = "Human"
            self.running = True
            self.agent = None
            self.current_score = 0.0
            self.game_over = False
            self.paused = False
            self.ai_status = "idle"
            # Reset environment and return the new state
            return self.env.reset()
        if not self.running and self.btn_ai.collidepoint(pos):
            # 顯示訓練配置對話框
            self.training_dialog = TrainingDialog(self.width, self.height)
            self.show_training_dialog = True
            return None
        if self.btn_board.collidepoint(pos):
            if self.mode == "Board":
                self.mode = "Menu"
            else:
                self.running = False
                self.paused = False
                self.game_over = False
                self.mode = "Board"
            return None

        # 清除排行榜按鈕（只在排行榜模式下才有效）
        if self.mode == "Board" and self.btn_clear_board is not None:
            if self.btn_clear_board.collidepoint(pos):
                # 清除所有排行榜紀錄
                self.leaderboard = []
                self._save_scores()
                print("✅ 已清除所有排行榜紀錄")
                return None

        if not self.running and self.btn_multi_view.collidepoint(pos):
            # 啟動多視窗觀看模式
            self._launch_multi_window_view()
            return None

    def _handle_save_training(self):
        if self.mode != "AI":
            return None

        print("📝 儲存訓練進度中...")
        self.ai_status = "saving"

        slot = self._active_slot()
        if slot is None or slot.trainer is None:
            print("⚠️ 目前沒有可儲存的訓練器")
            return None

        # 停止訓練視窗以避免與 checkpoint 寫入衝突
        if slot.training_window is not None:
            slot.training_window.stop()
            slot.training_window = None

        if slot.stop_event is not None:
            slot.stop_event.set()
        if slot.trainer_thread is not None and slot.trainer_thread.is_alive():
            slot.trainer_thread.join(timeout=5.0)

        trainer_ref = slot.trainer
        checkpoint_path = None
        base_iteration = self.training_iterations or self.n or 0
        iteration = int(max(0, base_iteration))

        if trainer_ref is not None:
            try:
                checkpoint_path = trainer_ref.save(iteration)
                print(f"✅ 已儲存訓練檔案: {checkpoint_path}")
            except Exception as err:
                print(f"⚠️ 儲存訓練檔案失敗: {err}")

        if trainer_ref is not None or iteration > 0 or self.ai_round > 0:
            self._save_training_meta(iteration, self.ai_round)

        self._stop_algorithm_training(wait=True)

        # 返回主選單狀態
        self.running = False
        self.mode = "Menu"
        self.ai_status = "saved"
        self.agent_ready = False
        self.btn_save = None
        self.current_score = 0.0

        try:
            new_state = self.env.reset()
        except Exception:
            new_state = None

        if checkpoint_path is None and trainer_ref is not None:
            print("提示: 未能寫入 checkpoint，請檢查檔案權限或磁碟空間。")

        return new_state

    def _handle_init_training(self, algorithm_key: Optional[str] = None):
        """初始化指定演算法的訓練資料（刪除模型和訓練進度）

        Args:
            algorithm_key: 要初始化的演算法鍵值。如果為 None，則使用當前活躍的演算法。
        """
        # 確定要初始化的演算法
        if algorithm_key is None:
            slot = self._active_slot()
            if slot is None:
                print("⚠️ 沒有選擇的演算法")
                return None
            algorithm_key = self.ai_manager.active_key
        else:
            slot = self.ai_manager.state(algorithm_key)
            if slot is None:
                print(f"⚠️ 找不到演算法: {algorithm_key}")
                return None

        desc = self.ai_manager.descriptor(algorithm_key)
        algo_name = desc.name if desc else algorithm_key

        print(f"�️ 初始化 {algo_name} 訓練資料（刪除所有進度）...")

        if self.starting_ai:
            print("AI 訓練初始化中，請稍候完成後再試。")
            return None

        if self.running and self.mode == "Human":
            print("請先結束人類遊戲模式，再進行訓練初始化。")
            return None

        # 如果這是當前活躍的演算法，更新狀態顯示
        if algorithm_key == self.ai_manager.active_key:
            self.ai_status = "resetting"
            self.agent_ready = False
            self.last_ai_action = None
            self.last_ai_action_prob = 0.0
            self.last_ai_value = 0.0

        # 如果正在訓練這個演算法，停止訓練
        if slot.trainer_thread is not None and slot.trainer_thread.is_alive():
            print(f"⏸️ 停止 {algo_name} 的訓練...")
            self._stop_algorithm_training(algorithm_key, wait=True)

        # 如果當前在 AI 模式使用這個演算法，返回選單
        if (
            self.mode == "AI"
            and self.running
            and algorithm_key == self.ai_manager.active_key
        ):
            self.running = False
            self.current_score = 0.0
            try:
                self.env.reset()
            except Exception:
                pass
            self.mode = "Menu"

        self.agent = None

        # 刪除訓練進度
        slot.iterations = 0
        slot.ai_round = 0
        slot.viewer_round = 0
        slot.loss_history = {"policy": [], "value": [], "entropy": [], "total": []}

        # 刪除模型檔案
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            import glob

            # 刪除這個演算法的所有 checkpoint 檔案
            # 支援兩種命名格式：
            # {algorithm_key}_checkpoint_*.pt 和 checkpoint_*.pt
            patterns = [
                os.path.join(checkpoint_dir, f"{algorithm_key}_checkpoint_*.pt"),
                os.path.join(
                    checkpoint_dir, "checkpoint_*.pt"
                ),  # 舊格式（如果是當前活躍演算法）
            ]

            deleted_count = 0
            # 如果這是當前活躍的演算法，也刪除沒有前綴的檔案
            if algorithm_key == self.ai_manager.active_key:
                for pattern in patterns:
                    files = glob.glob(pattern)
                    for f in files:
                        try:
                            os.remove(f)
                            print(f"  ✓ 已刪除: {os.path.basename(f)}")
                            deleted_count += 1
                        except Exception as e:
                            print(f"  ✗ 無法刪除 {os.path.basename(f)}: {e}")
            else:
                # 如果不是活躍演算法，只刪除有前綴的檔案
                files = glob.glob(patterns[0])
                for f in files:
                    try:
                        os.remove(f)
                        print(f"  ✓ 已刪除: {os.path.basename(f)}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ✗ 無法刪除 {os.path.basename(f)}: {e}")

            if deleted_count == 0:
                print(f"  ℹ️ 沒有找到 {algo_name} 的 checkpoint 檔案")

        # 更新狀態
        slot.status = "idle"
        if algorithm_key == self.ai_manager.active_key:
            self.ai_status = "idle"

        print(f"✅ {algo_name} 初始化完成，所有訓練資料已清除。")
        return None

    def _draw_loss_plot(self, x, y, w, h):
        """Draw multiple loss series (policy, value, entropy, total) into panel area.

        x,y are screen coordinates, w,h are dimensions.
        """
        surf = pygame.Surface((w, h))
        surf.fill((20, 20, 30))
        # draw dark background
        pygame.draw.rect(surf, (24, 24, 28), (0, 0, w, h))

        # determine max length among series (thread-safe copy)
        slot = self._active_slot()
        if slot is None:
            self.screen.blit(surf, (x, y))
            return
        with self._lock:
            lh_copy = {k: list(v) for k, v in slot.loss_history.items()}

        algo_name = slot.descriptor.name if slot.descriptor else "訓練"
        title = self.font.render(f"{algo_name} Loss 趨勢", True, (190, 200, 240))
        surf.blit(title, (8, 6))

        max_len = 0
        for v in lh_copy.values():
            if v:
                max_len = max(max_len, len(v))

        if max_len < 2:
            # draw a hint text
            hint = self.font.render("等待 Loss 數據中...", True, (150, 150, 150))
            surf.blit(hint, (8, title.get_height() + 10))
            self.screen.blit(surf, (x, y))
            return

        N = min(max_len, w)
        series_colors = {
            "policy": (200, 80, 80),
            "value": (80, 200, 120),
            "entropy": (120, 120, 200),
            "total": (220, 220, 80),
        }

        # 繪製圖例
        legend_y = title.get_height() + 10
        for name, color in series_colors.items():
            label = self.font.render(name.capitalize(), True, color)
            surf.blit(label, (w - label.get_width() - 5, legend_y))
            legend_y += 20

        for name, color in series_colors.items():
            seq = list(lh_copy.get(name, []))
            if not seq:
                continue
            seq = seq[-N:]
            mx = max(seq) if seq else 0
            mn = min(seq) if seq else 0
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
        """Export actor weights to TensorBoard or save a local numpy snapshot.

        Best-effort: leverage get_weight_matrix() when the network exposes it.
        """
        try:
            w = None
            if (
                self.agent is not None
                and hasattr(self.agent, "net")
                and hasattr(self.agent.net, "get_weight_matrix")
            ):
                w = self.agent.net.get_weight_matrix()
            elif hasattr(self.env, "net") and hasattr(
                self.env.net, "get_weight_matrix"
            ):
                w = self.env.net.get_weight_matrix()

            if w is None:
                # nothing to export
                return

            # try tensorboard
            try:
                import numpy as _np
                from torch.utils.tensorboard import SummaryWriter

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
                    entries = []
                    for item in data:
                        name = "AI"
                        score = 0
                        iteration = None
                        note = None
                        if isinstance(item, dict):
                            name = item.get("name", name)
                            score = item.get("score", score)
                            iteration = item.get("iteration", iteration)
                            note = item.get("note")
                        elif isinstance(item, (list, tuple)):
                            if len(item) >= 1:
                                name = item[0]
                            if len(item) >= 2:
                                score = item[1]
                            if len(item) >= 3:
                                candidate = item[2]
                                if isinstance(candidate, str):
                                    note = candidate
                                    digits = "".join(
                                        ch for ch in candidate if ch.isdigit()
                                    )
                                    if digits:
                                        try:
                                            iteration = int(digits)
                                        except Exception:
                                            iteration = None
                                else:
                                    iteration = candidate
                        try:
                            score = int(score)
                        except Exception:
                            continue
                        if iteration is not None:
                            try:
                                iteration = int(iteration)
                            except Exception:
                                iteration = None
                        if not note and isinstance(iteration, int):
                            note = f"(第{iteration:,}次訓練)"
                        entries.append(
                            {
                                "name": str(name),
                                "score": score,
                                "iteration": iteration,
                                "note": note,
                            }
                        )
                    if entries:
                        self.leaderboard = entries
                        max_iter = max(
                            (
                                e.get("iteration")
                                for e in entries
                                if isinstance(e.get("iteration"), int)
                            ),
                            default=-1,
                        )
                        if max_iter > self.training_iterations:
                            self.training_iterations = max_iter
                            self.n = self.training_iterations
            except Exception:
                # ignore malformed
                pass

    def _save_scores(self):
        p = os.path.join("checkpoints", "scores.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.leaderboard, f, ensure_ascii=False, indent=2)

    def _check_and_update_best_checkpoint(self, current_score, iteration_idx):
        """遊戲回合結束時，檢查是否打破記錄並立即更新 checkpoint_best.pt"""
        import shutil

        # 讀取歷史最高分
        historical_best = 0
        if self.leaderboard:
            historical_best = max(entry.get("score", 0) for entry in self.leaderboard)

        # 如果打破記錄
        if current_score > historical_best:
            # 找到最近的檢查點（訓練迭代是10的倍數）
            nearest_checkpoint_iter = (iteration_idx // 10) * 10
            checkpoint_path = os.path.join(
                "checkpoints", f"checkpoint_{nearest_checkpoint_iter}.pt"
            )
            best_path = os.path.join("checkpoints", "checkpoint_best.pt")

            # 如果檢查點存在，立即更新 checkpoint_best.pt
            if os.path.exists(checkpoint_path):
                try:
                    shutil.copy2(checkpoint_path, best_path)
                    print(
                        f"💎 立即更新最佳檢查點: checkpoint_best.pt "
                        f"(來源: checkpoint_{nearest_checkpoint_iter}.pt, "
                        f"遊戲回合 #{iteration_idx}, 分數: {current_score})"
                    )
                except Exception as e:
                    print(f"⚠️  更新最佳檢查點失敗: {e}")

    def _load_training_meta(self):
        path = os.path.join("checkpoints", "training_meta.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        algorithms_meta = data.get("algorithms")
        if isinstance(algorithms_meta, dict):
            for key, meta in algorithms_meta.items():
                try:
                    slot = self.ai_manager.state(key)
                except KeyError:
                    continue
                try:
                    slot.iterations = int(meta.get("iteration", slot.iterations))
                    slot.n = slot.iterations
                except Exception:
                    pass
                try:
                    slot.ai_round = int(meta.get("episodes", slot.ai_round))
                except Exception:
                    pass
                try:
                    slot.viewer_round = int(meta.get("viewer_round", slot.viewer_round))
                except Exception:
                    pass
        else:
            # backward compatibility with legacy schema
            try:
                last_it = int(data.get("last_iteration", self.training_iterations))
                self.training_iterations = max(self.training_iterations, last_it)
                self.n = self.training_iterations
            except Exception:
                pass

            try:
                total_eps = int(data.get("total_episodes", self.ai_round))
                self.ai_round = max(self.ai_round, total_eps)
            except Exception:
                pass

        try:
            stored_envs = int(data.get("vector_envs", self.vector_envs))
            self.vector_envs = max(1, stored_envs)
        except Exception:
            pass

        active_key = data.get("active")
        if isinstance(active_key, str):
            self._set_active_algorithm(active_key)

        self._sync_vector_env_index()

    def _save_training_meta(self, iteration: int, episodes: int) -> None:
        """暫存訓練統計資訊（向後兼容舊結構）。"""
        path = os.path.join("checkpoints", "training_meta.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        snapshot = {}
        for key, slot in self.ai_manager.agents_snapshot().items():
            snapshot[key] = {
                "iteration": int(max(0, slot.iterations)),
                "episodes": int(max(0, slot.ai_round)),
                "viewer_round": int(max(0, slot.viewer_round)),
            }

        payload = {
            "algorithms": snapshot,
            "vector_envs": int(max(1, self.vector_envs)),
            "active": self.ai_manager.active_key,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _latest_checkpoint(self):
        ckpt_dir = "checkpoints"
        best_iter = -1
        best_path = None
        try:
            for name in os.listdir(ckpt_dir):
                if not name.startswith("checkpoint_") or not name.endswith(".pt"):
                    continue
                try:
                    it_val = int(name[len("checkpoint_") : -3])
                except Exception:
                    continue
                if it_val > best_iter:
                    best_iter = it_val
                    best_path = os.path.join(ckpt_dir, name)
        except Exception:
            return None, None
        return best_path, (best_iter if best_iter >= 0 else None)

    def _refresh_training_counters(self):
        latest_path, latest_iter = self._latest_checkpoint()
        if isinstance(latest_iter, int) and latest_iter >= 0:
            if latest_iter > self.training_iterations:
                self.training_iterations = latest_iter
                self.n = latest_iter

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
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                elif event.type == pygame.KEYDOWN:
                    # ESC 鍵暫停/取消暫停
                    if (
                        event.key == pygame.K_ESCAPE
                        and self.running
                        and not self.game_over
                    ):
                        self.paused = not self.paused
                    # 空白鍵跳躍（只在遊戲進行中且未暫停時）
                    elif (
                        self.running
                        and self.mode == "Human"
                        and event.key == pygame.K_SPACE
                        and not self.paused
                        and not self.game_over
                    ):
                        self.human_jump = True
                    elif event.key in self.algorithm_hotkeys:
                        key = self.algorithm_hotkeys[event.key]
                        self._set_active_algorithm(key)

            # if not running (menu mode), only render and wait for user to click start
            if not self.running:
                # 檢查模式是否改變，需要重新計算佈局
                if self._last_layout_mode != (self.mode, self.running):
                    self._update_layout(self.width, self.height)
                    self._last_layout_mode = (self.mode, self.running)

                # render only - don't step the environment
                self.screen.fill(self.BG_COLOR)
                self._draw_algorithm_panel()  # 繪製左側演算法面板
                self.draw_playfield(s)
                self.draw_panel()
                # 繪製訓練對話框（如果顯示）
                if self.show_training_dialog and self.training_dialog is not None:
                    self.training_dialog.draw(self.screen)
                pygame.display.flip()
                self.clock.tick(self.FPS)
                continue

            # 檢查模式是否改變，需要重新計算佈局（運行中）
            if self._last_layout_mode != (self.mode, self.running):
                self._update_layout(self.width, self.height)
                self._last_layout_mode = (self.mode, self.running)

            # 如果遊戲暫停或結束，只渲染不更新
            if self.paused or self.game_over:
                self.screen.fill(self.BG_COLOR)
                # 只在非 AI 模式時顯示演算法面板
                if self.mode != "AI":
                    self._draw_algorithm_panel()
                self.draw_playfield(s)
                self.draw_panel()
                if self.paused:
                    self.draw_pause_dialog()
                elif self.game_over:
                    self.draw_game_over_dialog()
                pygame.display.flip()
                self.clock.tick(self.FPS)
                continue

            steps_this_frame = 1
            if self.mode == "AI":
                steps_this_frame = max(1, int(self.ai_speed_multiplier))

            for _ in range(steps_this_frame):
                if self.mode == "AI":
                    if self.agent is not None:
                        a, logp, value = self.agent.act(s)
                        next_s, r, done, info = self.env.step(a)

                        self.last_ai_action = a
                        prob = math.exp(logp) if logp > -10 else 0.0
                        self.last_ai_action_prob = max(0.0, min(1.0, prob))
                        self.last_ai_value = value
                        s = next_s
                    else:
                        self.last_ai_action = None
                        self.last_ai_action_prob = 0.0
                        self.last_ai_value = 0.0
                        s, r, done, _ = self.env.step(0)
                else:
                    if self.human_jump:
                        action = 1
                        self.human_jump = False
                    else:
                        action = 0
                    s, r, done, _ = self.env.step(action)

                try:
                    self.current_score += float(r)
                except Exception:
                    pass

                if not done:
                    continue

                # 檢查是否勝利（達到 99999 分）
                is_win = info.get("win", False)

                if self.mode == "Human":
                    self.game_over = True
                    entry_note = "🎉 通關！" if is_win else None
                    self.leaderboard.append(
                        {
                            "name": "人類",
                            "score": int(self.current_score),
                            "iteration": None,
                            "note": entry_note,
                        }
                    )
                    self.leaderboard = sorted(
                        self.leaderboard, key=lambda x: x["score"], reverse=True
                    )[:50]
                    try:
                        self._save_scores()
                    except Exception:
                        pass
                else:
                    slot = self._active_slot()
                    algo_name = slot.descriptor.name if slot is not None else "AI"
                    name = f"AI-{algo_name}"
                    score = int(self.current_score)
                    iteration_idx = int(self.training_iterations)

                    # 檢查是否勝利
                    is_win = info.get("win", False)
                    if is_win:
                        note_text = f"🎉 通關！{algo_name} 第{iteration_idx:,}次訓練"
                    else:
                        note_text = f"{algo_name} 第{iteration_idx:,}次訓練"

                    self.leaderboard.append(
                        {
                            "name": name,
                            "score": score,
                            "iteration": iteration_idx,
                            "note": note_text,
                        }
                    )
                    self.leaderboard = sorted(
                        self.leaderboard, key=lambda x: x["score"], reverse=True
                    )[:50]
                    try:
                        self._save_scores()
                    except Exception:
                        pass

                    # 檢查是否打破歷史記錄，立即更新 checkpoint_best.pt
                    try:
                        self._check_and_update_best_checkpoint(score, iteration_idx)
                    except Exception:
                        pass

                    win_text = " 🎉 通關！" if is_win else ""
                    print(
                        f"AI 回合 {self.viewer_round + 1} 結束，"
                        f"分數: {score}{win_text} "
                        f"(第{iteration_idx:,}次訓練)"
                    )

                    self.screen.fill(self.BG_COLOR)
                    # AI 模式下不顯示演算法面板
                    self.draw_playfield(s)
                    self.draw_panel()
                    pygame.display.flip()
                    pygame.time.wait(300)

                    self.current_score = 0.0
                    self.viewer_round += 1
                    s = self.env.reset()

                break

            # render
            self.screen.fill(self.BG_COLOR)
            # AI 訓練模式下不顯示演算法面板
            if self.mode != "AI":
                self._draw_algorithm_panel()
            self.draw_playfield(s)
            self.draw_panel()
            if self.game_over:
                self.draw_game_over_dialog()
            pygame.display.flip()
            self.clock.tick(self.FPS)

        # 清理：停止所有訓練
        self.ai_manager.stop_all()
        self.agent = None
        self.agent_ready = False
        self.ai_status = "idle"

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
