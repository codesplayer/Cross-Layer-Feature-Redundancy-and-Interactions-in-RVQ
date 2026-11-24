"""MOSS 模型加载与显存管理。"""

from __future__ import annotations

from .util import load_model  # noqa: F401
from .config import default_model_path


def load_moss(device: str = "npu:0"):
    """加载本地 MOSS-Audio-Tokenizer，返回 (model, sr)。"""
    return load_model(default_model_path(), device)


def free_gpu() -> None:
    """释放 CUDA / NPU 缓存（克隆 / 大模型切换之间调用）。"""
    from .npu_parallel import free_device

    free_device()
