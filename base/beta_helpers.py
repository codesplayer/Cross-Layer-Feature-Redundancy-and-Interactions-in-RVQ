"""β 扫参小工具（自 residual_observability_beta_n3 精简）。"""

from __future__ import annotations


def parse_betas(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


def beta_tag(b: float) -> str:
    return f"beta{b:g}".replace(".", "p")
