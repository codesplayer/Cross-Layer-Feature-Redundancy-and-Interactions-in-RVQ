"""从已有 per_clip 重算 Anchor 三方法 hold-out FAR / TPR / AUC。

去掉 random_500 定标集（默认 D:/data/.../random_500/split.json），
阈值用 COMPARE_FAR01 冻结 T；负类=干净无印。

例：
  python -m P1.experiments.recompute_anchor_holdout \\
    --result-root D:/KimiData/kimi/Workspaces/P1/exp/P1/anchor_attack_three_n1500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from base.anchor_attacks import DEFAULT_ANCHOR_KEYS

# COMPARE_FAR01 冻结阈值
T_WM = 1e-12
T_AS = 0.18587467074494227
T_ABA = 0.08451842905452735
T_SLIDE = 0.3238710454914386

SPLIT_DEFAULT = Path(
    r"D:\data\codec_an\anas\band\attack_layer_gap_n1500_calib\random_500\split.json"
)
RESULT_DEFAULT = Path(
    r"D:\KimiData\kimi\Workspaces\P1\exp\P1\anchor_attack_three_n1500"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def auc_mw(pos: np.ndarray, neg: np.ndarray) -> float | None:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg, dtype=np.float64).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return None
    all_s = np.concatenate([pos, neg])
    n_pos, n_neg = int(pos.size), int(neg.size)
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty(all_s.size, dtype=np.float64)
    ranks[order] = np.arange(1, all_s.size + 1, dtype=np.float64)
    sorted_s = all_s[order]
    i = 0
    while i < sorted_s.size:
        j = i
        while j + 1 < sorted_s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    return float(
        (float(ranks[labels == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def load_calib(split_json: Path) -> tuple[set[str], dict]:
    if not split_json.is_file():
        raise SystemExit(f"missing split: {split_json}")
    sp = _load(split_json)
    names = set(sp.get("calib_names") or [])
    meta = {
        "split": str(split_json),
        "calib_n": int(sp.get("calib_n") or len(names)),
        "calib_seed": sp.get("calib_seed"),
        "scenario": sp.get("scenario"),
        "corpus_counts": sp.get("corpus_counts"),
    }
    return names, meta


def load_method(per_clip: Path) -> list[dict]:
    rows = []
    for fp in sorted(per_clip.glob("*.json")):
        if fp.name.startswith("."):
            continue
        rows.append(_load(fp))
    return rows


def holdout(rows: list[dict], calib: set[str]) -> list[dict]:
    return [r for r in rows if r.get("name") not in calib]


def _finite(xs) -> np.ndarray:
    out = []
    for x in xs:
        if x is None:
            continue
        v = float(x)
        if np.isfinite(v):
            out.append(v)
    return np.asarray(out, dtype=np.float64)


def eval_baseline(hold: list[dict], T: float, method: str) -> dict:
    neg = _finite([r.get("clean_orig_prob") for r in hold])
    rows = []
    for key in DEFAULT_ANCHOR_KEYS:
        pos = []
        for r in hold:
            blk = (r.get("attacks") or {}).get(key) or {}
            if "error" in blk:
                continue
            pos.append(blk.get("wm_prob"))
        pos_a = _finite(pos)
        rows.append(
            {
                "attack": key,
                "n": int(pos_a.size),
                "mean_wm": float(np.mean(pos_a)) if pos_a.size else None,
                "mean_orig": float(np.mean(neg)) if neg.size else None,
                "T": float(T),
                "far": float(np.mean(neg >= T)) if neg.size else None,
                "tpr": float(np.mean(pos_a >= T)) if pos_a.size else None,
                "auc": auc_mw(pos_a, neg) if pos_a.size and neg.size else None,
            }
        )
    return {
        "method": method,
        "n_files": None,
        "n_eval": len(hold),
        "T": float(T),
        "protocol": "far01_frozen_holdout",
        "neg": "clean_orig",
        "rows": rows,
    }


def _cascade_score(aba: np.ndarray, slide: np.ndarray, t_aba: float, t_slide: float) -> np.ndarray:
    """软级联分数：max(aba/T_aba, slide/T_slide)；≥1 等价于 aba∨slide 硬判决。"""
    ta = max(float(t_aba), 1e-12)
    ts = max(float(t_slide), 1e-12)
    return np.maximum(np.asarray(aba, float) / ta, np.asarray(slide, float) / ts)


def eval_band(hold: list[dict], t_aba: float, t_slide: float) -> dict:
    neg_a = _finite([r.get("clean_orig_aba") for r in hold])
    neg_s = _finite([r.get("clean_orig_slide") for r in hold])
    n_neg = min(neg_a.size, neg_s.size)
    neg_a, neg_s = neg_a[:n_neg], neg_s[:n_neg]
    neg_cas = _cascade_score(neg_a, neg_s, t_aba, t_slide) if n_neg else np.asarray([])
    cas_far = float(np.mean(neg_cas >= 1.0)) if n_neg else None
    rows = []
    for key in DEFAULT_ANCHOR_KEYS:
        pos_a, pos_s = [], []
        for r in hold:
            blk = (r.get("attacks") or {}).get(key) or {}
            if "error" in blk:
                continue
            pos_a.append(blk.get("aba_embed"))
            pos_s.append(blk.get("slide_v3"))
        pa, ps = _finite(pos_a), _finite(pos_s)
        n = min(pa.size, ps.size)
        pa, ps = pa[:n], ps[:n]
        pos_cas = _cascade_score(pa, ps, t_aba, t_slide) if n else np.asarray([])
        cas_tpr = float(np.mean(pos_cas >= 1.0)) if n else None
        rows.append(
            {
                "attack": key,
                "n": int(n),
                "mean_wm_aba": float(np.mean(pa)) if n else None,
                "mean_wm_slide": float(np.mean(ps)) if n else None,
                "mean_orig_aba": float(np.mean(neg_a)) if neg_a.size else None,
                "mean_orig_slide": float(np.mean(neg_s)) if neg_s.size else None,
                "T_aba": float(t_aba),
                "T_slide": float(t_slide),
                "far_aba": float(np.mean(neg_a >= t_aba)) if neg_a.size else None,
                "tpr_aba": float(np.mean(pa >= t_aba)) if n else None,
                "auc_aba": auc_mw(pa, neg_a) if n and neg_a.size else None,
                "far_slide": float(np.mean(neg_s >= t_slide)) if neg_s.size else None,
                "tpr_slide": float(np.mean(ps >= t_slide)) if n else None,
                "auc_slide": auc_mw(ps, neg_s) if n and neg_s.size else None,
                "cascade_far": cas_far,
                "cascade_tpr": cas_tpr,
                "auc_cascade": auc_mw(pos_cas, neg_cas) if n and n_neg else None,
            }
        )
    return {
        "method": "band",
        "n_eval": len(hold),
        "T_aba": float(t_aba),
        "T_slide": float(t_slide),
        "protocol": "far01_frozen_holdout",
        "neg": "clean_orig",
        "decision": "aba>=T_aba OR slide>=T_slide",
        "cascade_score": "max(aba/T_aba, slide/T_slide)",
        "rows": rows,
    }


def write_md(out_root: Path, by_method: dict, split_meta: dict) -> None:
    def pct(x):
        return "—" if x is None else f"{100 * float(x):.2f}%"

    def fmt(x, nd=4):
        return "—" if x is None else f"{float(x):.{nd}f}"

    n_eval = next(
        (by_method[m]["n_eval"] for m in ("wavmark", "audioseal", "band") if m in by_method),
        None,
    )
    lines = [
        "# Anchor 十类攻击 @ far01（hold-out，去掉 random_500）",
        "",
        f"- split: `{split_meta.get('split')}`  "
        f"scenario={split_meta.get('scenario')} seed={split_meta.get('calib_seed')}  "
        f"calib_n={split_meta.get('calib_n')} → n_eval≈{n_eval}",
        f"- 冻结阈值：WavMark T={T_WM:g}；AudioSeal T={T_AS:.6f}；"
        f"band (T_aba, T_slide)=({T_ABA:.6f}, {T_SLIDE:.6f})",
        "- 负类：干净无印；band 硬判决 `aba≥T_aba ∨ slide≥T_slide`（无 wave）",
        "- AUC：Mann-Whitney；级联软分数 `max(aba/T_aba, slide/T_slide)`（≥1 ⇔ 硬级联）",
        "",
        "## 主表：三方法 TPR / FAR / AUC",
        "",
        "| attack | WavMark TPR/FAR/AUC | AudioSeal TPR/FAR/AUC | band ABA TPR/FAR/AUC | band 级联 TPR/FAR/AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    wm = {r["attack"]: r for r in (by_method.get("wavmark") or {}).get("rows") or []}
    as_ = {r["attack"]: r for r in (by_method.get("audioseal") or {}).get("rows") or []}
    bd = {r["attack"]: r for r in (by_method.get("band") or {}).get("rows") or []}
    for key in DEFAULT_ANCHOR_KEYS:
        w, a, b = wm.get(key, {}), as_.get(key, {}), bd.get(key, {})
        lines.append(
            f"| {key} | {pct(w.get('tpr'))}/{pct(w.get('far'))}/{fmt(w.get('auc'))} | "
            f"{pct(a.get('tpr'))}/{pct(a.get('far'))}/{fmt(a.get('auc'))} | "
            f"{pct(b.get('tpr_aba'))}/{pct(b.get('far_aba'))}/{fmt(b.get('auc_aba'))} | "
            f"{pct(b.get('cascade_tpr'))}/{pct(b.get('cascade_far'))}/{fmt(b.get('auc_cascade'))} |"
        )
    lines += [
        "",
        "## band 细表：ABA / slide / 级联",
        "",
        "| attack | ABA TPR | ABA FAR | ABA AUC | slide TPR | slide FAR | slide AUC | 级联 TPR | 级联 FAR | 级联 AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in DEFAULT_ANCHOR_KEYS:
        b = bd.get(key, {})
        lines.append(
            f"| {key} | {pct(b.get('tpr_aba'))} | {pct(b.get('far_aba'))} | {fmt(b.get('auc_aba'))} | "
            f"{pct(b.get('tpr_slide'))} | {pct(b.get('far_slide'))} | {fmt(b.get('auc_slide'))} | "
            f"{pct(b.get('cascade_tpr'))} | {pct(b.get('cascade_far'))} | {fmt(b.get('auc_cascade'))} |"
        )
    lines.append("")
    path = out_root / "COMPARE_ANCHOR_FAR01_HOLDOUT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result-root", type=str, default=str(RESULT_DEFAULT))
    p.add_argument("--split-json", type=str, default=str(SPLIT_DEFAULT))
    p.add_argument("--t-wm", type=float, default=T_WM)
    p.add_argument("--t-as", type=float, default=T_AS)
    p.add_argument("--t-aba", type=float, default=T_ABA)
    p.add_argument("--t-slide", type=float, default=T_SLIDE)
    args = p.parse_args()

    root = Path(args.result_root)
    calib, split_meta = load_calib(Path(args.split_json))
    by_method = {}
    for method, T in (("wavmark", args.t_wm), ("audioseal", args.t_as), ("band", None)):
        pc = root / method / "per_clip"
        if not pc.is_dir():
            print(f"skip missing {pc}", flush=True)
            continue
        rows = load_method(pc)
        hold = holdout(rows, calib)
        print(
            f"{method}: n_files={len(rows)} n_calib_in={sum(1 for r in rows if r.get('name') in calib)} "
            f"n_eval={len(hold)}",
            flush=True,
        )
        if method == "band":
            summary = eval_band(hold, args.t_aba, args.t_slide)
        else:
            summary = eval_baseline(hold, float(T), method)
        summary["n_files"] = len(rows)
        summary["n_calib_removed"] = len(rows) - len(hold)
        summary["split"] = split_meta
        _dump(root / method / "summary_far01_holdout.json", summary)
        by_method[method] = summary

    payload = {
        "split": split_meta,
        "thresholds": {
            "wavmark": float(args.t_wm),
            "audioseal": float(args.t_as),
            "T_aba": float(args.t_aba),
            "T_slide": float(args.t_slide),
        },
        "methods": by_method,
    }
    _dump(root / "all_summaries_holdout.json", payload)
    write_md(root, by_method, split_meta)
    print("wrote", root / "all_summaries_holdout.json", flush=True)


if __name__ == "__main__":
    main()
