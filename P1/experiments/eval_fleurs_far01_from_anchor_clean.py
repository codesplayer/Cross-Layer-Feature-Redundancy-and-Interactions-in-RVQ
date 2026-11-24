"""用 anchor_attack_three_n1500 的 **clean 负类** 做 FAR01 定标，再评 fleurs_2040。

定标
----
从 ``{calib-root}/{method}/per_clip`` 的 ``clean_orig_*`` 取阈值，使标定集 FAR≤1%
（``roc_threshold_at_far``；band 分别定 T_aba / T_slide）。

评估集
----
- ``mild_attack``：宽松攻击 × {audioseal, wavmark, band}
- ``wm_n2000``：Anchor 十类 × 三方法
- ``cs_n2000``：crop/speed 盲搜（band；用 best.aba/slide 级联）

每行至少报 **FAR↓ / TPR↑ / AUC↑**，并附 **95% CI**
（率：Wilson；AUC：Hanley–McNeil；``--boot-n >0`` 时额外 bootstrap 核对）。

例::

  python -m P1.experiments.eval_fleurs_far01_from_anchor_clean \\
    --calib-root D:/KimiData/kimi/Workspaces/P1/exp/P1/anchor_attack_three_n1500 \\
    --fleurs-root D:/KimiData/kimi/Workspaces/P1/exp/P1/fleurs_2040 \\
    --out-dir D:/KimiData/kimi/Workspaces/P1/exp/P1/fleurs_2040/far01_from_anchor_clean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from base.anchor_attacks import DEFAULT_ANCHOR_KEYS
from base.mild_attacks import DEFAULT_MILD_KEYS
from base.util import dump_json
from P1.experiments.compute_layer_gap_auc import (
    auc_mw,
    cascade_score,
    roc_threshold_at_far,
)
from P1.experiments.recompute_anchor_holdout import _finite

TARGET_FAR = 0.01
CI_LEVEL = 0.95
Z95 = 1.959963984540054
BOOT_SEED = 0
# 运行时由 main 写入；auc_pack 读取
_RUNTIME = {"boot_n": 0}

CALIB_DEFAULT = Path(r"D:\KimiData\kimi\Workspaces\P1\exp\P1\anchor_attack_three_n1500")
FLEURS_DEFAULT = Path(r"D:\KimiData\kimi\Workspaces\P1\exp\P1\fleurs_2040")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_per_clip(per_clip: Path) -> list[dict]:
    rows = []
    for fp in sorted(per_clip.glob("*.json")):
        if fp.name.startswith("."):
            continue
        try:
            rows.append(_load(fp))
        except Exception:
            continue
    return rows


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """Wilson score interval for proportion k/n."""
    if n <= 0:
        return None, None
    phat = k / n
    den = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / den
    half = (z / den) * float(np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def rate_pack(hits: int, n: int) -> dict:
    if n <= 0:
        return {"value": None, "n": 0, "hits": 0, "ci95": [None, None]}
    lo, hi = wilson_ci(int(hits), int(n))
    return {
        "value": float(hits / n),
        "n": int(n),
        "hits": int(hits),
        "ci95": [lo, hi],
    }


def hanley_mcneil_auc_ci(
    auc: float, n_pos: int, n_neg: int, z: float = Z95
) -> tuple[float | None, float | None, float | None]:
    """Hanley–McNeil (1982) SE → 95% CI；返回 (se, lo, hi)。"""
    if n_pos <= 0 or n_neg <= 0 or auc is None or not np.isfinite(auc):
        return None, None, None
    a = float(np.clip(auc, 1e-9, 1.0 - 1e-9))
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    se2 = (
        a * (1.0 - a)
        + (n_pos - 1) * (q1 - a * a)
        + (n_neg - 1) * (q2 - a * a)
    ) / (n_pos * n_neg)
    se = float(np.sqrt(max(se2, 0.0)))
    lo = float(np.clip(a - z * se, 0.0, 1.0))
    hi = float(np.clip(a + z * se, 0.0, 1.0))
    return se, lo, hi


def bootstrap_auc_ci(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    n_boot: int,
    seed: int = BOOT_SEED,
    alpha: float = 1.0 - CI_LEVEL,
) -> tuple[float | None, float | None]:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg, dtype=np.float64).reshape(-1)
    if n_boot <= 0 or pos.size == 0 or neg.size == 0:
        return None, None
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_boot)):
        p = pos[rng.integers(0, pos.size, size=pos.size)]
        n = neg[rng.integers(0, neg.size, size=neg.size)]
        a = auc_mw(p, n)
        if a is not None and np.isfinite(a):
            vals.append(float(a))
    if not vals:
        return None, None
    lo, hi = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _stable_seed(*parts, base: int = BOOT_SEED) -> int:
    h = int(base) & 0xFFFFFFFF
    for p in parts:
        for ch in str(p):
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return int(h)


def auc_pack(pos: np.ndarray, neg: np.ndarray, *, seed: int = BOOT_SEED) -> dict:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg, dtype=np.float64).reshape(-1)
    point = auc_mw(pos, neg)
    se, lo, hi = (
        hanley_mcneil_auc_ci(float(point), int(pos.size), int(neg.size))
        if point is not None
        else (None, None, None)
    )
    boot_n = int(_RUNTIME.get("boot_n") or 0)
    blo = bhi = None
    if boot_n > 0 and point is not None:
        blo, bhi = bootstrap_auc_ci(pos, neg, n_boot=boot_n, seed=seed)
    return {
        "value": point,
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
        "se_hanley": se,
        "ci95": [lo, hi],
        "ci_method": "hanley_mcneil",
        "ci95_bootstrap": [blo, bhi] if boot_n > 0 else None,
        "boot_n": boot_n,
    }


def fmt_pct(x, nd: int = 2) -> str:
    return "—" if x is None else f"{100 * float(x):.{nd}f}%"


def fmt_num(x, nd: int = 4) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def fmt_rate_ci(pack: dict | None) -> str:
    if not pack or pack.get("value") is None:
        return "—"
    lo, hi = (pack.get("ci95") or [None, None])[:2]
    return f"{fmt_pct(pack['value'])} [{fmt_pct(lo)}, {fmt_pct(hi)}]"


def fmt_auc_ci(pack: dict | None) -> str:
    if not pack or pack.get("value") is None:
        return "—"
    lo, hi = (pack.get("ci95") or [None, None])[:2]
    return f"{fmt_num(pack['value'])} [{fmt_num(lo)}, {fmt_num(hi)}]"


# ---------------------------------------------------------------------------
# Calibrate from anchor clean
# ---------------------------------------------------------------------------


def calibrate_baseline(rows: list[dict], *, target_far: float = TARGET_FAR) -> dict:
    neg = _finite([r.get("clean_orig_prob") for r in rows])
    pos = _finite([r.get("clean_wm_prob") for r in rows])
    hit = roc_threshold_at_far(pos, neg, target_far)
    return {
        "T": hit.get("T"),
        "calib_far": hit.get("far"),
        "calib_tpr_clean_wm": hit.get("tpr"),
        "n_neg": int(neg.size),
        "n_pos_clean_wm": int(pos.size),
        "target_far": float(target_far),
    }


def calibrate_band(rows: list[dict], *, target_far: float = TARGET_FAR) -> dict:
    neg_a = _finite([r.get("clean_orig_aba") for r in rows])
    neg_s = _finite([r.get("clean_orig_slide") for r in rows])
    pos_a = _finite([r.get("clean_wm_aba") for r in rows])
    pos_s = _finite([r.get("clean_wm_slide") for r in rows])
    ha = roc_threshold_at_far(pos_a, neg_a, target_far)
    hs = roc_threshold_at_far(pos_s, neg_s, target_far)
    t_aba, t_slide = ha.get("T"), hs.get("T")
    if t_aba is None or t_slide is None:
        raise SystemExit("band FAR01 calibration failed (empty clean scores)")
    neg_cas = cascade_score(neg_a, neg_s, t_aba, t_slide)
    pos_cas = cascade_score(pos_a, pos_s, t_aba, t_slide)
    return {
        "T_aba": float(t_aba),
        "T_slide": float(t_slide),
        "calib_far_aba": ha.get("far"),
        "calib_tpr_aba_clean_wm": ha.get("tpr"),
        "calib_far_slide": hs.get("far"),
        "calib_tpr_slide_clean_wm": hs.get("tpr"),
        "calib_cascade_far": float(np.mean(neg_cas >= 1.0)) if neg_cas.size else None,
        "calib_cascade_tpr_clean_wm": float(np.mean(pos_cas >= 1.0)) if pos_cas.size else None,
        "n_neg": int(min(neg_a.size, neg_s.size)),
        "n_pos_clean_wm": int(min(pos_a.size, pos_s.size)),
        "target_far": float(target_far),
    }


def calibrate_all(calib_root: Path, *, target_far: float = TARGET_FAR) -> dict:
    out: dict = {"source": str(calib_root), "target_far": float(target_far), "methods": {}}
    for method in ("wavmark", "audioseal", "band"):
        pc = calib_root / method / "per_clip"
        if not pc.is_dir():
            print(f"calib skip missing {pc}", flush=True)
            continue
        rows = load_per_clip(pc)
        if method == "band":
            cal = calibrate_band(rows, target_far=target_far)
        else:
            cal = calibrate_baseline(rows, target_far=target_far)
        cal["method"] = method
        cal["n_files"] = len(rows)
        out["methods"][method] = cal
        if method == "band":
            print(
                f"calib {method}: T_aba={cal['T_aba']:.6f} T_slide={cal['T_slide']:.6f} "
                f"far_aba={cal['calib_far_aba']} far_slide={cal['calib_far_slide']} "
                f"cas_far={cal['calib_cascade_far']} n={cal['n_neg']}",
                flush=True,
            )
        else:
            print(
                f"calib {method}: T={cal['T']} far={cal['calib_far']} n={cal['n_neg']}",
                flush=True,
            )
    if not out["methods"]:
        raise SystemExit(f"no methods calibrated under {calib_root}")
    return out


# ---------------------------------------------------------------------------
# Evaluate one attack suite / cs
# ---------------------------------------------------------------------------


def _attack_keys(rows: list[dict], preferred: tuple[str, ...] | list[str]) -> list[str]:
    seen: set[str] = set()
    for r in rows:
        seen.update((r.get("attacks") or {}).keys())
    keys = [k for k in preferred if k in seen]
    for k in sorted(seen):
        if k not in keys:
            keys.append(k)
    return keys


def eval_baseline_suite(
    rows: list[dict],
    T: float,
    *,
    attack_keys: list[str],
    seed: int = BOOT_SEED,
) -> dict:
    neg = _finite([r.get("clean_orig_prob") for r in rows])
    far_hits = int(np.sum(neg >= float(T))) if neg.size else 0
    far = rate_pack(far_hits, int(neg.size))
    table = []
    for key in attack_keys:
        pos = []
        for r in rows:
            blk = (r.get("attacks") or {}).get(key) or {}
            if "error" in blk:
                continue
            pos.append(blk.get("wm_prob"))
        pos_a = _finite(pos)
        tpr_hits = int(np.sum(pos_a >= float(T))) if pos_a.size else 0
        table.append(
            {
                "attack": key,
                "n_pos": int(pos_a.size),
                "n_neg": int(neg.size),
                "mean_wm": float(np.mean(pos_a)) if pos_a.size else None,
                "mean_orig": float(np.mean(neg)) if neg.size else None,
                "T": float(T),
                "far": far,
                "tpr": rate_pack(tpr_hits, int(pos_a.size)),
                "auc": auc_pack(pos_a, neg, seed=_stable_seed(seed, "base", key)),
            }
        )
    return {
        "decision": "score >= T",
        "T": float(T),
        "n_eval": len(rows),
        "neg": "clean_orig_prob (eval set)",
        "rows": table,
    }


def eval_band_suite(
    rows: list[dict],
    t_aba: float,
    t_slide: float,
    *,
    attack_keys: list[str],
    seed: int = BOOT_SEED,
) -> dict:
    neg_a = _finite([r.get("clean_orig_aba") for r in rows])
    neg_s = _finite([r.get("clean_orig_slide") for r in rows])
    n_neg = min(neg_a.size, neg_s.size)
    neg_a, neg_s = neg_a[:n_neg], neg_s[:n_neg]
    neg_cas = cascade_score(neg_a, neg_s, t_aba, t_slide) if n_neg else np.asarray([])
    far_hits = int(np.sum(neg_cas >= 1.0)) if n_neg else 0
    far = rate_pack(far_hits, int(n_neg))
    # also single-channel FAR for reference
    far_aba = rate_pack(int(np.sum(neg_a >= t_aba)) if n_neg else 0, int(n_neg))
    far_slide = rate_pack(int(np.sum(neg_s >= t_slide)) if n_neg else 0, int(n_neg))

    table = []
    for key in attack_keys:
        pos_a, pos_s = [], []
        for r in rows:
            blk = (r.get("attacks") or {}).get(key) or {}
            if "error" in blk:
                continue
            pos_a.append(blk.get("aba_embed"))
            pos_s.append(blk.get("slide_v3"))
        pa, ps = _finite(pos_a), _finite(pos_s)
        n = min(pa.size, ps.size)
        pa, ps = pa[:n], ps[:n]
        pos_cas = cascade_score(pa, ps, t_aba, t_slide) if n else np.asarray([])
        tpr_hits = int(np.sum(pos_cas >= 1.0)) if n else 0
        tpr_cas = rate_pack(tpr_hits, int(n))
        tpr_aba = rate_pack(int(np.sum(pa >= t_aba)) if n else 0, int(n))
        tpr_slide = rate_pack(int(np.sum(ps >= t_slide)) if n else 0, int(n))
        auc_cas = auc_pack(pos_cas, neg_cas, seed=_stable_seed(seed, "band", key))
        auc_aba = auc_pack(pa, neg_a, seed=_stable_seed(seed, "aba", key))
        auc_slide = auc_pack(ps, neg_s, seed=_stable_seed(seed, "slide", key))
        table.append(
            {
                "attack": key,
                "n_pos": int(n),
                "n_neg": int(n_neg),
                "mean_wm_aba": float(np.mean(pa)) if n else None,
                "mean_wm_slide": float(np.mean(ps)) if n else None,
                "T_aba": float(t_aba),
                "T_slide": float(t_slide),
                # 级联（主字段 far/tpr/auc 兼容旧表）
                "far": far,
                "tpr": tpr_cas,
                "auc": auc_cas,
                "far_cascade": far,
                "tpr_cascade": tpr_cas,
                "auc_cascade": auc_cas,
                # 单通道 ABA / slide
                "far_aba": far_aba,
                "tpr_aba": tpr_aba,
                "auc_aba": auc_aba,
                "far_slide": far_slide,
                "tpr_slide": tpr_slide,
                "auc_slide": auc_slide,
            }
        )
    return {
        "decision": "aba>=T_aba OR slide>=T_slide  (soft: max(aba/T_aba,slide/T_slide)>=1)",
        "T_aba": float(t_aba),
        "T_slide": float(t_slide),
        "n_eval": len(rows),
        "neg": "clean_orig aba/slide (eval set)",
        "rows": table,
    }


def eval_cs_blind(
    rows: list[dict],
    t_aba: float,
    t_slide: float,
    *,
    seed: int = BOOT_SEED,
) -> dict:
    """cs_n2000：pos=blind best aba/slide 级联；neg=clean_orig。"""
    neg_a = _finite([r.get("clean_orig_aba") for r in rows])
    neg_s = _finite([r.get("clean_orig_slide") for r in rows])
    n_neg = min(neg_a.size, neg_s.size)
    neg_a, neg_s = neg_a[:n_neg], neg_s[:n_neg]
    neg_cas = cascade_score(neg_a, neg_s, t_aba, t_slide) if n_neg else np.asarray([])
    far_cas = rate_pack(int(np.sum(neg_cas >= 1.0)) if n_neg else 0, int(n_neg))
    far_aba = rate_pack(int(np.sum(neg_a >= t_aba)) if n_neg else 0, int(n_neg))
    far_slide = rate_pack(int(np.sum(neg_s >= t_slide)) if n_neg else 0, int(n_neg))

    table = []
    for ak in ("crop_mid", "speed"):
        pos_a, pos_s, pos_ch = [], [], []
        for r in rows:
            best = ((r.get(ak) or {}).get("blind_wm") or {}).get("best") or {}
            if not best:
                continue
            pos_a.append(best.get("aba_embed"))
            pos_s.append(best.get("slide_v3"))
            pos_ch.append(best.get("channel_max"))
        pa, ps = _finite(pos_a), _finite(pos_s)
        n = min(pa.size, ps.size)
        pa, ps = pa[:n], ps[:n]
        pos_cas = cascade_score(pa, ps, t_aba, t_slide) if n else np.asarray([])
        pch = _finite(pos_ch)[:n]
        tpr_cas = rate_pack(int(np.sum(pos_cas >= 1.0)) if n else 0, int(n))
        tpr_aba = rate_pack(int(np.sum(pa >= t_aba)) if n else 0, int(n))
        tpr_slide = rate_pack(int(np.sum(ps >= t_slide)) if n else 0, int(n))
        # channel_max：与 blind 脚本一致，阈值取 min(T_aba,T_slide)
        t_ch = min(float(t_aba), float(t_slide))
        tpr_ch = rate_pack(int(np.sum(pch >= t_ch)) if pch.size else 0, int(pch.size))
        neg_soft = (
            np.maximum(neg_a / max(t_aba, 1e-12), neg_s / max(t_slide, 1e-12))
            if n_neg
            else np.asarray([])
        )
        auc_cas = auc_pack(pos_cas, neg_cas, seed=_stable_seed(seed, "cs", ak))
        table.append(
            {
                "attack": ak,
                "n_pos": int(n),
                "n_neg": int(n_neg),
                "mean_channel_max": float(np.mean(pch)) if pch.size else None,
                "far": far_cas,
                "tpr": tpr_cas,
                "auc": auc_cas,
                "far_cascade": far_cas,
                "tpr_cascade": tpr_cas,
                "auc_cascade": auc_cas,
                "far_aba": far_aba,
                "tpr_aba": tpr_aba,
                "auc_aba": auc_pack(pa, neg_a, seed=_stable_seed(seed, "cs_aba", ak)),
                "far_slide": far_slide,
                "tpr_slide": tpr_slide,
                "auc_slide": auc_pack(ps, neg_s, seed=_stable_seed(seed, "cs_slide", ak)),
                "tpr_channel_max": tpr_ch,
                "auc_channel_max": auc_pack(
                    pch, neg_soft, seed=_stable_seed(seed, "cs_ch", ak)
                ),
            }
        )
    return {
        "decision": "blind best: aba>=T_aba OR slide>=T_slide",
        "T_aba": float(t_aba),
        "T_slide": float(t_slide),
        "n_eval": len(rows),
        "neg": "clean_orig aba/slide (eval set)",
        "rows": table,
    }


def eval_suite_root(
    suite_root: Path,
    calib: dict,
    *,
    preferred_attacks: list[str] | tuple[str, ...],
    seed: int = BOOT_SEED,
) -> dict:
    methods_out = {}
    for method in ("wavmark", "audioseal", "band"):
        pc = suite_root / method / "per_clip"
        if not pc.is_dir():
            print(f"eval skip missing {pc}", flush=True)
            continue
        rows = load_per_clip(pc)
        keys = _attack_keys(rows, preferred_attacks)
        cal = (calib.get("methods") or {}).get(method) or {}
        if method == "band":
            if cal.get("T_aba") is None:
                print(f"eval skip {method}: no calib T", flush=True)
                continue
            summary = eval_band_suite(
                rows, float(cal["T_aba"]), float(cal["T_slide"]), attack_keys=keys, seed=seed
            )
        else:
            if cal.get("T") is None:
                print(f"eval skip {method}: no calib T", flush=True)
                continue
            summary = eval_baseline_suite(rows, float(cal["T"]), attack_keys=keys, seed=seed)
        summary["method"] = method
        summary["n_files"] = len(rows)
        methods_out[method] = summary
        print(f"eval {suite_root.name}/{method}: n={len(rows)} attacks={len(keys)}", flush=True)
    return {
        "suite": suite_root.name,
        "root": str(suite_root),
        "methods": methods_out,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fta(r: dict, far_k: str, tpr_k: str, auc_k: str) -> str:
    """一行：FAR / TPR / AUC，均带 95% CI。"""
    return (
        f"{fmt_rate_ci(r.get(far_k))} / {fmt_rate_ci(r.get(tpr_k))} / {fmt_auc_ci(r.get(auc_k))}"
    )


def write_suite_md(path: Path, title: str, suite: dict, calib: dict) -> None:
    lines = [
        f"# {title}",
        "",
        f"- eval root: `{suite.get('root')}`",
        f"- calib root: `{calib.get('source')}`  target_FAR={calib.get('target_far')}",
        "- thresholds from **anchor clean_orig** @ FAR01；评测 FAR 用本集 clean_orig",
        "- **95% CI**：FAR/TPR=Wilson；AUC=Hanley–McNeil",
        "- band：同时报 **ABA 单通道 / slide 单通道 / 级联**",
        "",
    ]
    cm = calib.get("methods") or {}
    if "wavmark" in cm:
        lines.append(
            f"- WavMark T={cm['wavmark'].get('T')}  "
            f"calib_FAR={fmt_pct(cm['wavmark'].get('calib_far'))}"
        )
    if "audioseal" in cm:
        lines.append(
            f"- AudioSeal T={fmt_num(cm['audioseal'].get('T'), 6)}  "
            f"calib_FAR={fmt_pct(cm['audioseal'].get('calib_far'))}"
        )
    if "band" in cm:
        b = cm["band"]
        lines.append(
            f"- band (T_aba, T_slide)=({fmt_num(b.get('T_aba'), 6)}, {fmt_num(b.get('T_slide'), 6)})  "
            f"calib_FAR_aba={fmt_pct(b.get('calib_far_aba'))}  "
            f"calib_FAR_slide={fmt_pct(b.get('calib_far_slide'))}  "
            f"calib_cascade_FAR={fmt_pct(b.get('calib_cascade_far'))}"
        )

    methods = suite.get("methods") or {}
    wm = {r["attack"]: r for r in (methods.get("wavmark") or {}).get("rows") or []}
    as_ = {r["attack"]: r for r in (methods.get("audioseal") or {}).get("rows") or []}
    bd = {r["attack"]: r for r in (methods.get("band") or {}).get("rows") or []}
    keys = []
    for src in (wm, as_, bd):
        for k in src:
            if k not in keys:
                keys.append(k)

    lines += [
        "",
        "## 主表：三方法 FAR↓ / TPR↑ / AUC↑（95% CI）",
        "",
        "单元格格式：`FAR [95%CI] / TPR [95%CI] / AUC [95%CI]`；band 列为**级联**。",
        "",
        "| attack | WavMark | AudioSeal | band 级联 |",
        "|---|---|---|---|",
    ]
    for key in keys:
        w, a, b = wm.get(key, {}), as_.get(key, {}), bd.get(key, {})
        lines.append(
            f"| {key} | {_fta(w, 'far', 'tpr', 'auc')} | "
            f"{_fta(a, 'far', 'tpr', 'auc')} | "
            f"{_fta(b, 'far_cascade', 'tpr_cascade', 'auc_cascade') if b.get('far_cascade') is not None else _fta(b, 'far', 'tpr', 'auc')} |"
        )

    if bd:
        lines += [
            "",
            "## band 细表：ABA / slide / 级联（均含 95% CI）",
            "",
            "| attack | ABA FAR↓ | ABA TPR↑ | ABA AUC↑ | "
            "slide FAR↓ | slide TPR↑ | slide AUC↑ | "
            "级联 FAR↓ | 级联 TPR↑ | 级联 AUC↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key in keys:
            b = bd.get(key) or {}
            if not b:
                continue
            lines.append(
                f"| {key} | {fmt_rate_ci(b.get('far_aba'))} | {fmt_rate_ci(b.get('tpr_aba'))} | "
                f"{fmt_auc_ci(b.get('auc_aba'))} | {fmt_rate_ci(b.get('far_slide'))} | "
                f"{fmt_rate_ci(b.get('tpr_slide'))} | {fmt_auc_ci(b.get('auc_slide'))} | "
                f"{fmt_rate_ci(b.get('far_cascade') or b.get('far'))} | "
                f"{fmt_rate_ci(b.get('tpr_cascade') or b.get('tpr'))} | "
                f"{fmt_auc_ci(b.get('auc_cascade') or b.get('auc'))} |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path, flush=True)


def write_cs_md(path: Path, cs: dict, calib: dict) -> None:
    bcal = (calib.get("methods") or {}).get("band") or {}
    lines = [
        "# fleurs cs_n2000 @ FAR01（定标自 anchor clean）",
        "",
        f"- eval: `{cs.get('root')}`",
        f"- calib: `{calib.get('source')}`",
        f"- band (T_aba, T_slide)=({fmt_num(bcal.get('T_aba'), 6)}, {fmt_num(bcal.get('T_slide'), 6)})",
        "- blind best 的 **ABA / slide 单通道 + 级联** vs clean_orig",
        "- **95% CI**：FAR/TPR=Wilson；AUC=Hanley–McNeil",
        "",
        "## 级联 + 单通道",
        "",
        "| attack | ABA FAR↓ | ABA TPR↑ | ABA AUC↑ | "
        "slide FAR↓ | slide TPR↑ | slide AUC↑ | "
        "级联 FAR↓ | 级联 TPR↑ | 级联 AUC↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in (cs.get("rows") or []):
        lines.append(
            f"| {r.get('attack')} | {fmt_rate_ci(r.get('far_aba'))} | {fmt_rate_ci(r.get('tpr_aba'))} | "
            f"{fmt_auc_ci(r.get('auc_aba'))} | {fmt_rate_ci(r.get('far_slide'))} | "
            f"{fmt_rate_ci(r.get('tpr_slide'))} | {fmt_auc_ci(r.get('auc_slide'))} | "
            f"{fmt_rate_ci(r.get('far_cascade') or r.get('far'))} | "
            f"{fmt_rate_ci(r.get('tpr_cascade') or r.get('tpr'))} | "
            f"{fmt_auc_ci(r.get('auc_cascade') or r.get('auc'))} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path, flush=True)


def write_index_md(out_dir: Path, calib: dict, suites: dict, cs: dict | None) -> None:
    lines = [
        "# fleurs_2040 FAR01（定标：anchor_attack_three_n1500 clean）",
        "",
        f"- calib: `{calib.get('source')}`  target_FAR={calib.get('target_far')}",
        "- 指标：FAR↓ / TPR↑ / AUC↑ + **95% CI**（Wilson / Hanley–McNeil）",
        "- band：ABA 单通道 / slide 单通道 / 级联 均报告",
        "",
        "## 定标阈值",
        "",
    ]
    for m, cal in (calib.get("methods") or {}).items():
        if m == "band":
            lines.append(
                f"- **band**: T_aba={fmt_num(cal.get('T_aba'), 6)}, "
                f"T_slide={fmt_num(cal.get('T_slide'), 6)}  "
                f"(calib FAR aba/slide/cascade="
                f"{fmt_pct(cal.get('calib_far_aba'))}/"
                f"{fmt_pct(cal.get('calib_far_slide'))}/"
                f"{fmt_pct(cal.get('calib_cascade_far'))})"
            )
        else:
            lines.append(
                f"- **{m}**: T={cal.get('T') if m == 'wavmark' else fmt_num(cal.get('T'), 6)}  "
                f"calib_FAR={fmt_pct(cal.get('calib_far'))}"
            )
    lines += ["", "## 结果文件", ""]
    for name in suites:
        lines.append(f"- `{name}/COMPARE_FAR01.md`")
        lines.append(f"- `{name}/summary_far01.json`")
    if cs is not None:
        lines.append("- `cs_n2000/COMPARE_FAR01.md`")
        lines.append("- `cs_n2000/summary_far01.json`")
    lines.append("- `thresholds_far01.json`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="FAR01 calib from anchor clean → fleurs eval + CI")
    p.add_argument("--calib-root", type=str, default=str(CALIB_DEFAULT))
    p.add_argument("--fleurs-root", type=str, default=str(FLEURS_DEFAULT))
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="默认 {fleurs-root}/far01_from_anchor_clean",
    )
    p.add_argument("--target-far", type=float, default=TARGET_FAR)
    p.add_argument(
        "--boot-n",
        type=int,
        default=0,
        help=">0 时额外算 AUC bootstrap CI（主 CI 仍为 Hanley–McNeil）",
    )
    p.add_argument("--seed", type=int, default=BOOT_SEED)
    p.add_argument(
        "--suites",
        type=str,
        default="mild_attack,wm_n2000,cs_n2000",
        help="逗号分隔；cs_n2000 单独 band 盲搜",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _RUNTIME["boot_n"] = max(0, int(args.boot_n))

    calib_root = Path(args.calib_root)
    fleurs = Path(args.fleurs_root)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (fleurs / "far01_from_anchor_clean")
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = calibrate_all(calib_root, target_far=float(args.target_far))
    dump_json(out_dir / "thresholds_far01.json", calib)

    suites_wanted = [x.strip() for x in str(args.suites).split(",") if x.strip()]
    suite_results = {}
    cs_result = None

    for name in suites_wanted:
        if name == "cs_n2000":
            cs_root = fleurs / "cs_n2000"
            pc = cs_root / "per_clip"
            if not pc.is_dir():
                print(f"skip missing {pc}", flush=True)
                continue
            bcal = (calib.get("methods") or {}).get("band") or {}
            if bcal.get("T_aba") is None:
                raise SystemExit("need band calib for cs_n2000")
            rows = load_per_clip(pc)
            cs_result = eval_cs_blind(
                rows, float(bcal["T_aba"]), float(bcal["T_slide"]), seed=int(args.seed)
            )
            cs_result["root"] = str(cs_root)
            cs_result["suite"] = "cs_n2000"
            cs_result["n_files"] = len(rows)
            sub = out_dir / "cs_n2000"
            sub.mkdir(parents=True, exist_ok=True)
            dump_json(sub / "summary_far01.json", cs_result)
            write_cs_md(sub / "COMPARE_FAR01.md", cs_result, calib)
            continue

        root = fleurs / name
        if name == "mild_attack":
            preferred = list(DEFAULT_MILD_KEYS)
            title = "fleurs mild_attack @ FAR01（定标自 anchor clean）"
        elif name == "wm_n2000":
            preferred = list(DEFAULT_ANCHOR_KEYS)
            title = "fleurs wm_n2000（Anchor 十类）@ FAR01（定标自 anchor clean）"
        else:
            preferred = list(DEFAULT_ANCHOR_KEYS) + list(DEFAULT_MILD_KEYS)
            title = f"fleurs {name} @ FAR01（定标自 anchor clean）"

        suite = eval_suite_root(root, calib, preferred_attacks=preferred, seed=int(args.seed))
        suite_results[name] = suite
        sub = out_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        dump_json(sub / "summary_far01.json", suite)
        write_suite_md(sub / "COMPARE_FAR01.md", title, suite, calib)

    dump_json(
        out_dir / "all_summaries_far01.json",
        {"thresholds": calib, "suites": suite_results, "cs_n2000": cs_result},
    )
    write_index_md(out_dir, calib, suite_results, cs_result)
    print(f"DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
