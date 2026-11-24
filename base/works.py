"""三条工作的嵌入↔检测配对（禁止跨工作串用）。"""

from __future__ import annotations

WORKS = {
    "P1": {
        "name": "冗余归因水印",
        "package": "P1",
        "embed": "P1.embed.embed_aba / base.embed_wav",
        "detect": "band_zq_of → detect_p1_cosine",
    },
    "P2": {
        "name": "补偿水印 / 抗变速",
        "package": "P2",
        "embed": "embed_p2(kind='static'|'time_pn')",
        "detect": "detect_p2",
    },
    "P3_clone": {
        "name": "抗克隆（二值符号）",
        "package": "P3",
        "embed": "embed_sign / embed_sign_multilayer；b=keyed_bits",
        "detect": "detect_sign / LayerNull",
    },
    "P3_probe": {
        "name": "信息流仪器（非水印系统）",
        "package": "P3",
        "embed": "embed_pn",
        "detect": "tap_stat",
    },
}
