"""P1 嵌入：稀有 ABA 整层替换。"""

from __future__ import annotations

import numpy as np

from base.rare_tokens import ensure_rare_table, layer_aba
from base.util import embed_wav


def embed_aba(
    model,
    wav: np.ndarray,
    *,
    layer: int = 9,
    table: dict | None = None,
    table_path: str | None = None,
    period: int = 3,
) -> dict:
    """在指定层写入 ABA；返回 embed_wav 的字典（含 wav/mod/plan）。"""
    if table is None:
        table = ensure_rare_table(table_path) if table_path else ensure_rare_table()
    return embed_wav(
        model,
        wav,
        table,
        high_start=int(layer),
        high_end=int(layer),
        layers=[int(layer)],
        period=int(period),
    )


def key_template(model, table: dict | None = None, layer: int = 9, T: int = 64, period: int = 3):
    """密钥侧 band 模板（检测用）。"""
    from base.residual_feat import aba_embed_template
    from base.rare_tokens import ensure_rare_table

    table = table or ensure_rare_table()
    return aba_embed_template(
        model, int(T), int(layer), int(layer), table, layers=[int(layer)], period=int(period)
    )
