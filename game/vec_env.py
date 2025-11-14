"""
向量化環境包裝器
使用 multiprocessing 實現真正的並行環境收集
"""

import multiprocessing as mp
from typing import List, Optional

import numpy as np

from game.environment import GameEnv


def _worker(remote, parent_remote, env_fn):
    """
    子進程工作函數
    接收命令並執行環境操作
    """
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_state":
                remote.send(env.get_state())
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close() if hasattr(env, "close") else None


class SubprocVecEnv:
    """
    使用子進程的向量化環境
    每個環境在獨立的進程中運行，實現真正的並行
    """

    def __init__(self, env_fns: List[callable], context: Optional[str] = None):
        """
        Args:
            env_fns: 環境工廠函數列表
            context: multiprocessing 上下文 ('spawn', 'fork', 'forkserver')
        """
        self.n_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        # 使用 spawn 上下文避免 Windows/PyTorch 問題
        if context is None:
            context = "spawn"
        ctx = mp.get_context(context)

        # 創建管道和進程
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        self.processes = [
            ctx.Process(target=_worker, args=(work_remote, remote, env_fn), daemon=True)
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]

        for p in self.processes:
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # 獲取環境狀態空間大小
        self.remotes[0].send(("reset", None))
        first_obs = self.remotes[0].recv()
        self.observation_shape = (
            first_obs.shape if hasattr(first_obs, "shape") else (len(first_obs),)
        )

    def step_async(self, actions):
        """非同步發送動作到所有環境"""
        if self.waiting:
            raise RuntimeError("step_async 已被調用，請先調用 step_wait")

        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        """等待所有環境完成 step 並返回結果"""
        if not self.waiting:
            raise RuntimeError("請先調用 step_async")

        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.array(rews), np.array(dones), list(infos)

    def step(self, actions):
        """同步 step：發送動作並等待結果"""
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        """重置所有環境"""
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def close(self):
        """關閉所有環境和進程"""
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                remote.recv()
            self.waiting = False

        for remote in self.remotes:
            remote.send(("close", None))

        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        self.closed = True

    def __len__(self):
        return self.n_envs

    def __del__(self):
        if not self.closed:
            self.close()


def make_vec_envs(n_envs: int = 4, context: Optional[str] = None) -> SubprocVecEnv:
    """
    創建向量化環境的便捷函數

    Args:
        n_envs: 環境數量
        context: multiprocessing 上下文

    Returns:
        SubprocVecEnv 實例
    """
    env_fns = [lambda: GameEnv() for _ in range(n_envs)]
    return SubprocVecEnv(env_fns, context=context)
