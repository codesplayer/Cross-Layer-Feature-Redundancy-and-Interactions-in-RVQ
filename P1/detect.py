"""P1 检测：band_zq 与密钥模板余弦。"""

from __future__ import annotations

from base.detect_shared import detect_p1_cosine
from base.residual_feat import band_zq_of


def detect_aba(model, wav, template, layer: int = 9) -> dict:
    feat = band_zq_of(model, wav, int(layer), int(layer))
    return detect_p1_cosine(feat, template)
