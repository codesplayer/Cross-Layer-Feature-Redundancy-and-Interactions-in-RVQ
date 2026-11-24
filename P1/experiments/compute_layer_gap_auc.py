"""对 attack_layer_gap_n1500 计算 band ABA / 级联 的 AUC（及 FAR/TPR @ far01）。

默认：profile=band_L9；random_500 hold-out；负类=干净无印 orig；
级联软分 max(aba/T_aba, slide/T_slide)。

例：
  python -m P1.experiments.compute_layer_gap_auc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT_DEFAULT = Path(r"D:\data\codec_an\eval\codec_an\exp_band\attack_layer_gap_n1500")
SPLIT_DEFAULT = Path(
    r"D:\data\codec_an\anas\band\attack_layer_gap_n1500_calib\random_500\split.json"
)
PROFILE_DEFAULT = "band_L9"
TARGET_FAR = 0.01

# 与 fair 主表一致的攻击顺序；clean 放最前
ATTACK_ORDER = [
    "clean",
    "resample_16k",
    "aac_128k",
    "mp3_128k",
    "reencode",
    "resample_48k",
    "resample_44100",
    "crop_mid",
    "noise",
    "bandpass",
    "boost",
    "duck",
    "echo",
    "highpass",
    "lowpass",
    "pink_noise",
    "white_noise",
    "smooth",
    "speed",
]


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _dump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def roc_threshold_at_far(pos: np.ndarray, neg: np.ndarray, target_far: float) -> dict:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg, dtype=np.float64).reshape(-1)
    if neg.size == 0:
        return {"T": None, "far": None, "tpr": None}
    cands = sorted(set(float(x) for x in np.unique(neg)) | {float(np.max(neg)) + 1e-12})
    best = None
    for th in cands:
        far = float(np.mean(neg >= th))
        if far <= target_far + 1e-12:
            tpr = float(np.mean(pos >= th)) if pos.size else None
            if best is None or (
                tpr is not None
                and (
                    tpr > best["tpr"] + 1e-15
                    or (abs(tpr - best["tpr"]) <= 1e-15 and th > best["T"])
                )
            ):
                best = {"T": float(th), "far": far, "tpr": tpr}
    if best is None:
        th = float(np.max(neg)) + 1e-12
        best = {
            "T": th,
            "far": float(np.mean(neg >= th)),
            "tpr": float(np.mean(pos >= th)) if pos.size else None,
        }
    return best


def cascade_score(aba, slide, t_aba, t_slide) -> np.ndarray:
    ta = max(float(t_aba), 1e-12)
    ts = max(float(t_slide), 1e-12)
    return np.maximum(np.asarray(aba, float) / ta, np.asarray(slide, float) / ts)


def extract_pair(rec: dict, profile: str, attack: str) -> tuple[float, float] | None:
    """返回 (aba, slide)；clean → wm 干净分；攻击 → attacks[atk].scores。"""
    prof = (rec.get("profiles") or {}).get(profile) or {}
    if attack == "clean":
        sc = ((prof.get("wm") or {}).get("scores")) or {}
    else:
        blk = ((prof.get("attacks") or {}).get(attack)) or {}
        sc = blk.get("scores") or {}
        if not sc:
            return None
    if "aba_embed" not in sc or "slide_v3" not in sc:
        return None
    return float(sc["aba_embed"]), float(sc["slide_v3"])


def extract_orig(rec: dict, profile: str) -> tuple[float, float] | None:
    sc = (((rec.get("profiles") or {}).get(profile) or {}).get("orig") or {}).get("scores") or {}
    if "aba_embed" not in sc or "slide_v3" not in sc:
        return None
    return float(sc["aba_embed"]), float(sc["slide_v3"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(ROOT_DEFAULT))
    p.add_argument("--split-json", type=str, default=str(SPLIT_DEFAULT))
    p.add_argument("--profile", type=str, default=PROFILE_DEFAULT)
    p.add_argument("--no-holdout", action="store_true", help="全量 1500，不去定标")
    args = p.parse_args()

    root = Path(args.root)
    per_clip = root / "per_clip"
    files = sorted(p for p in per_clip.glob("*.json") if not p.name.startswith("."))
    rows_all = [_load(fp) for fp in files]
    for r, fp in zip(rows_all, files):
        r.setdefault("name", fp.stem)

    split_meta = {}
    calib: set[str] = set()
    if not args.no_holdout and Path(args.split_json).is_file():
        sp = _load(Path(args.split_json))
        calib = set(sp.get("calib_names") or [])
        split_meta = {
            "split": str(args.split_json),
            "scenario": sp.get("scenario"),
            "calib_n": int(sp.get("calib_n") or len(calib)),
            "calib_seed": sp.get("calib_seed"),
        }
    hold = [r for r in rows_all if r.get("name") not in calib] if calib else list(rows_all)
    calib_rows = [r for r in rows_all if r.get("name") in calib] if calib else []

    profile = args.profile

    # 定标集干净 orig/wm → far01 T
    def collect_clean(rows, which: str):
        a, s = [], []
        for r in rows:
            if which == "orig":
                pair = extract_orig(r, profile)
            else:
                pair = extract_pair(r, profile, "clean")
            if pair is None:
                continue
            a.append(pair[0])
            s.append(pair[1])
        return np.asarray(a, float), np.asarray(s, float)

    calib_use = calib_rows if calib_rows else hold
    pos_a_c, pos_s_c = collect_clean(calib_use, "wm")
    neg_a_c, neg_s_c = collect_clean(calib_use, "orig")
    far_a = roc_threshold_at_far(pos_a_c, neg_a_c, TARGET_FAR)
    far_s = roc_threshold_at_far(pos_s_c, neg_s_c, TARGET_FAR)
    t_aba = float(far_a["T"]) if far_a["T"] is not None else 0.3027
    t_slide = float(far_s["T"]) if far_s["T"] is not None else 0.3298

    neg_a, neg_s = collect_clean(hold, "orig")
    n_neg = min(neg_a.size, neg_s.size)
    neg_a, neg_s = neg_a[:n_neg], neg_s[:n_neg]
    neg_cas = cascade_score(neg_a, neg_s, t_aba, t_slide)

    # 攻击列表：固定顺序 ∩ 实际出现
    seen = set()
    for r in hold:
        atk = ((r.get("profiles") or {}).get(profile) or {}).get("attacks") or {}
        seen.update(atk.keys())
    attacks = [a for a in ATTACK_ORDER if a == "clean" or a in seen]
    for a in sorted(seen):
        if a not in attacks:
            attacks.append(a)

    table = []
    for atk in attacks:
        pos_a, pos_s = [], []
        for r in hold:
            pair = extract_pair(r, profile, atk)
            if pair is None:
                continue
            pos_a.append(pair[0])
            pos_s.append(pair[1])
        pa, ps = np.asarray(pos_a, float), np.asarray(pos_s, float)
        n = min(pa.size, ps.size)
        if n == 0:
            continue
        pa, ps = pa[:n], ps[:n]
        pos_cas = cascade_score(pa, ps, t_aba, t_slide)
        table.append(
            {
                "attack": atk,
                "n": int(n),
                "mean_wm_aba": float(np.mean(pa)),
                "mean_wm_slide": float(np.mean(ps)),
                "mean_orig_aba": float(np.mean(neg_a)) if n_neg else None,
                "mean_orig_slide": float(np.mean(neg_s)) if n_neg else None,
                "tpr_aba": float(np.mean(pa >= t_aba)),
                "far_aba": float(np.mean(neg_a >= t_aba)) if n_neg else None,
                "auc_aba": auc_mw(pa, neg_a) if n_neg else None,
                "tpr_slide": float(np.mean(ps >= t_slide)),
                "far_slide": float(np.mean(neg_s >= t_slide)) if n_neg else None,
                "auc_slide": auc_mw(ps, neg_s) if n_neg else None,
                "cascade_tpr": float(np.mean(pos_cas >= 1.0)),
                "cascade_far": float(np.mean(neg_cas >= 1.0)) if n_neg else None,
                "auc_cascade": auc_mw(pos_cas, neg_cas) if n_neg else None,
            }
        )

    payload = {
        "root": str(root),
        "profile": profile,
        "n_files": len(rows_all),
        "n_eval": len(hold),
        "n_calib_removed": len(rows_all) - len(hold),
        "split": split_meta,
        "thresholds_far01": {
            "T_aba": t_aba,
            "T_slide": t_slide,
            "calib_far_aba": far_a.get("far"),
            "calib_tpr_aba": far_a.get("tpr"),
            "calib_far_slide": far_s.get("far"),
            "calib_tpr_slide": far_s.get("tpr"),
            "note": "far01 frozen on calib clean orig/wm; cascade score max(aba/T_aba,slide/T_slide)",
        },
        "neg": "clean_orig",
        "rows": table,
    }
    out_json = root / f"AUC_{profile}_holdout.json"
    _dump(out_json, payload)

    def pct(x):
        return "—" if x is None else f"{100 * float(x):.2f}%"

    def fmt(x, nd=4):
        return "—" if x is None else f"{float(x):.{nd}f}"

    lines = [
        f"# {profile}：ABA / 级联 AUC（attack_layer_gap_n1500）",
        "",
        f"- root: `{root}`",
        f"- n_files={payload['n_files']}  n_eval={payload['n_eval']}  "
        f"n_calib_removed={payload['n_calib_removed']}",
        f"- split: `{split_meta.get('split', 'none')}`  "
        f"scenario={split_meta.get('scenario')}",
        f"- far01 冻结：T_aba={t_aba:.6f}  T_slide={t_slide:.6f}",
        "- 负类：干净无印 orig；AUC=Mann-Whitney；"
        "级联软分 `max(aba/T_aba, slide/T_slide)`（≥1⇔硬级联）",
        "- 分数来源：`profiles[profile].(wm|attacks[*]).scores`（raw，非 crop/speed 特化）",
        "",
        "| attack | ABA TPR | ABA FAR | ABA AUC | 级联 TPR | 级联 FAR | 级联 AUC | slide AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['attack']} | {pct(r['tpr_aba'])} | {pct(r['far_aba'])} | {fmt(r['auc_aba'])} | "
            f"{pct(r['cascade_tpr'])} | {pct(r['cascade_far'])} | {fmt(r['auc_cascade'])} | "
            f"{fmt(r['auc_slide'])} |"
        )
    lines.append("")
    out_md = root / f"AUC_{profile}_holdout.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out_md, flush=True)
    print("wrote", out_json, flush=True)
    print(
        f"T_aba={t_aba:.6f} T_slide={t_slide:.6f} n_eval={len(hold)} n_attacks={len(table)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
