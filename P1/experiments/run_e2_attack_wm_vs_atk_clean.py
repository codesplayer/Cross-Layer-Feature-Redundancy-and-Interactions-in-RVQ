"""E2：只算「攻击下干净」检测分；攻击水印复用已有 mild/anchor 结果再合并。

本脚本只跑::

  score(attack(clean))   # 负类

已有攻击水印（正类）来自例如::

  mild_attack_three_n1500/{method}/per_clip
  anchor_attack_three_n1500/{method}/per_clip

目录::

  {out}/{mild|anchor}/{band|audioseal|wavmark}/per_clip/{stem}.json

合并评测::

  python -m P1.experiments.run_e2_attack_wm_vs_atk_clean --summary-only \\
    --out-dir /path/e2_atk_clean \\
    --wm-mild-root /path/mild_attack_three_n1500 \\
    --wm-anchor-root /path/anchor_attack_three_n1500

例（只跑干净攻击）::

  python -m P1.experiments.run_e2_attack_wm_vs_atk_clean \\
    --wav-dir /path/mix_1500 --out-dir /path/e2_atk_clean \\
    --suites mild,anchor --methods audioseal,wavmark,band \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from base.anchor_attacks import DEFAULT_ANCHOR_KEYS, build_anchor_attacks
from base.channels import P1_BAND_L9, profile_score_cfg, score_wav_once
from base.config import EXP_ROOT, default_mix_1500, default_model_path, default_table_path
from base.mild_attacks import DEFAULT_MILD_KEYS, build_mild_attacks
from base.moss_io import read_wav, set_seed
from base.npu_parallel import (
    add_npu_args,
    build_payloads,
    ensure_spawn,
    free_device,
    limit_cpu_threads,
    open_worker_log,
    parse_npu_ids,
    pin_npu,
    run_shards,
)
from base.rare_tokens import ensure_rare_table
from base.util import dump_json, load_model
from P1.baselines import (
    AS_DET_CANDIDATES,
    AS_GEN_CANDIDATES,
    WM_CKPT_CANDIDATES,
    audioseal_detect,
    disable_torch_compile_on_npu,
    first_file,
    load_audioseal,
    load_wavmark,
    wavmark_detect,
)
from P1.experiments.compute_layer_gap_auc import auc_mw, cascade_score, roc_threshold_at_far
from P1.experiments.eval_fleurs_far01_from_anchor_clean import rate_pack
from P1.experiments.run_anchor_attack_three_n1500 import list_clips, load_clips_from_manifest

OUT_STEM = "e2_attack_clean_detect"
_OUT_CANDS = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
    Path(r"E:\code\water_marks\result") / OUT_STEM,
    EXP_ROOT / OUT_STEM,
)
DEFAULT_OUT = next((p for p in _OUT_CANDS if p.parent.exists()), _OUT_CANDS[0])
TARGET_FAR = 0.01
METRIC_VERSION = 3  # clean-attack only


def suite_attack_keys(suite: str) -> list[str]:
    suite = str(suite).strip().lower()
    if suite == "mild":
        return list(DEFAULT_MILD_KEYS)
    if suite == "anchor":
        return list(DEFAULT_ANCHOR_KEYS)
    raise SystemExit(f"unknown suite={suite}; use mild|anchor")


def build_attacks_for_suite(suite: str, keys: list[str], *, moss_model, crop_keep, snr_db, speed_factor):
    suite = str(suite).strip().lower()
    if suite == "mild":
        return build_mild_attacks(
            keys, moss_model=moss_model, crop_keep=crop_keep, snr_db=snr_db, speed_factor=speed_factor
        )
    if suite == "anchor":
        return build_anchor_attacks(keys)
    raise SystemExit(f"unknown suite={suite}")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def clip_ok(path: Path, method: str) -> bool:
    if not path.is_file():
        return False
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if d.get("error") or d.get("metric_version") != METRIC_VERSION:
        return False
    if d.get("method") != method:
        return False
    atk = d.get("attacks") or {}
    if not isinstance(atk, dict) or not atk:
        return False
    for blk in atk.values():
        if not isinstance(blk, dict) or "error" in blk:
            continue
        if method == "band" and blk.get("clean_aba") is not None:
            return True
        if method != "band" and blk.get("clean_prob") is not None:
            return True
    return False


def process_band_clean(model, sr, table, stem, wav_path, attacks, *, suite: str) -> dict:
    wav = np.asarray(read_wav(str(wav_path), sr), dtype=np.float32).reshape(-1)
    cfg = profile_score_cfg(table, P1_BAND_L9)
    so = score_wav_once(model, wav, sr, cfg, n_peel=9, period=3)
    atk = {}
    for key, fn, meta in attacks:
        try:
            w_cl = fn(wav, int(sr))
            s_cl = score_wav_once(model, w_cl, sr, cfg, n_peel=9, period=3)
            atk[key] = {
                "clean_aba": float(s_cl["aba_embed"]),
                "clean_slide": float(s_cl["slide18"]),
                "meta": meta,
            }
        except Exception as e:
            atk[key] = {"error": f"{type(e).__name__}: {e}", "meta": meta}
    return {
        "name": stem,
        "method": "band",
        "attack_suite": suite,
        "metric_version": METRIC_VERSION,
        "clean_orig_aba": float(so["aba_embed"]),
        "clean_orig_slide": float(so["slide18"]),
        "attacks": atk,
        "sr": int(sr),
        "note": "only attack(clean); merge with existing attack(wm) for E2",
    }


def process_audioseal_clean(det, device, stem, wav_path, attacks, *, suite: str) -> dict:
    wav16 = np.asarray(read_wav(str(wav_path), 16000), dtype=np.float32).reshape(-1)
    p_orig = audioseal_detect(det, wav16, 16000, device)
    atk = {}
    for key, fn, meta in attacks:
        try:
            w_cl = fn(wav16, 16000)
            atk[key] = {
                "clean_prob": float(audioseal_detect(det, w_cl, 16000, device)),
                "meta": meta,
            }
        except Exception as e:
            atk[key] = {"error": f"{type(e).__name__}: {e}", "meta": meta}
    return {
        "name": stem,
        "method": "audioseal",
        "attack_suite": suite,
        "metric_version": METRIC_VERSION,
        "clean_orig_prob": float(p_orig),
        "attacks": atk,
        "sr": 16000,
        "note": "only attack(clean); merge with existing attack(wm) for E2",
    }


def process_wavmark_clean(model, stem, wav_path, attacks, *, suite: str) -> dict:
    import librosa

    wav_n, sr0 = librosa.load(str(wav_path), sr=None, mono=True)
    wav = np.asarray(wav_n, dtype=np.float32).reshape(-1)
    sr0 = int(sr0)
    d_o = wavmark_detect(model, wav, sr0, None)
    atk = {}
    for key, fn, meta in attacks:
        try:
            w_cl = fn(wav, sr0)
            d_cl = wavmark_detect(model, w_cl, sr0, None)
            atk[key] = {"clean_prob": float(d_cl["wm_prob"]), "meta": meta}
        except Exception as e:
            atk[key] = {"error": f"{type(e).__name__}: {e}", "meta": meta}
    return {
        "name": stem,
        "method": "wavmark",
        "attack_suite": suite,
        "metric_version": METRIC_VERSION,
        "clean_orig_prob": float(d_o["wm_prob"]),
        "attacks": atk,
        "sr": int(sr0),
        "note": "only attack(clean); merge with existing attack(wm) for E2",
    }


def _worker(payload: dict) -> dict:
    log_fp, log = open_worker_log(payload)
    wid = int(payload.get("worker_id", 0))
    method = str(payload["method"])
    suite = str(payload.get("attack_suite") or "mild").lower()
    device = str(payload["device"])
    out_pc = Path(payload["per_clip_dir"])
    resume = bool(payload.get("resume", True))
    n_ok = n_err = n_skip = 0
    try:
        limit_cpu_threads(int(payload.get("cpu_threads", 1) or 1))
        pin_npu(device)
        set_seed(int(payload.get("seed", 0)) + wid)

        moss_for_atk = None
        band_model = table = sr = None
        as_det = None
        wm_model = None
        attack_keys = list(payload.get("attack_keys") or [])
        if method == "band":
            band_model, sr = load_model(str(payload["moss_path"]), device)
            table = ensure_rare_table(str(payload["table"]))
            moss_for_atk = band_model
        elif method == "audioseal":
            disable_torch_compile_on_npu(device)
            _gen, as_det = load_audioseal(
                str(payload["as_gen"]), str(payload["as_det"]), device, nbits=int(payload.get("nbits", 16))
            )
            if suite == "mild" and "reencode" in attack_keys:
                moss_for_atk, _ = load_model(str(payload["moss_path"]), device)
        elif method == "wavmark":
            wm_model = load_wavmark(str(payload["wm_ckpt"]), device)
            if suite == "mild" and "reencode" in attack_keys:
                moss_for_atk, _ = load_model(str(payload["moss_path"]), device)
        else:
            raise SystemExit(f"unknown method {method}")

        attacks = build_attacks_for_suite(
            suite,
            attack_keys,
            moss_model=moss_for_atk,
            crop_keep=float(payload.get("crop_keep", 0.7)),
            snr_db=float(payload.get("snr_db", 20.0)),
            speed_factor=float(payload.get("speed_factor", 0.9)),
        )
        log(f"worker{wid} suite={suite} method={method} CLEAN-ONLY n={len(payload['items'])}")

        for i, (stem, path) in enumerate(payload["items"]):
            dest = out_pc / f"{stem}.json"
            if resume and clip_ok(dest, method):
                n_skip += 1
                n_ok += 1
                continue
            try:
                if method == "band":
                    rec = process_band_clean(band_model, sr, table, stem, path, attacks, suite=suite)
                elif method == "audioseal":
                    rec = process_audioseal_clean(as_det, device, stem, path, attacks, suite=suite)
                else:
                    rec = process_wavmark_clean(wm_model, stem, path, attacks, suite=suite)
                _write_json(dest, rec)
                n_ok += 1
            except Exception as e:
                n_err += 1
                log(f"ERR {stem}: {e}\n{traceback.format_exc()}")
                _write_json(
                    out_pc.parent / "errors" / f"{stem}.json",
                    {"name": stem, "error": str(e), "trace": traceback.format_exc()},
                )
            if (i + 1) % 20 == 0 or i == 0:
                log(f"progress {i+1}/{len(payload['items'])} ok={n_ok} skip={n_skip} err={n_err}")
        return {
            "worker_id": wid,
            "suite": suite,
            "method": method,
            "n_ok": n_ok,
            "n_skip": n_skip,
            "n_error": n_err,
        }
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        return {
            "worker_id": wid,
            "suite": suite,
            "method": method,
            "n_ok": n_ok,
            "n_error": n_err + 1,
            "error": str(e),
        }
    finally:
        free_device()
        if log_fp:
            log_fp.close()


def _finite(xs) -> np.ndarray:
    out = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if np.isfinite(v):
            out.append(v)
    return np.asarray(out, dtype=np.float64)


def load_json_dir(per_clip: Path) -> dict[str, dict]:
    """stem -> rec（兼容 name/stem 字段）。"""
    out = {}
    if not per_clip.is_dir():
        return out
    for fp in per_clip.glob("*.json"):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("error"):
            continue
        stem = str(d.get("name") or d.get("stem") or fp.stem)
        out[stem] = d
    return out


def _wm_scores_from_existing(blk: dict, method: str) -> tuple:
    """从 mild/anchor 已有 attack(wm) 块取正类分。"""
    if method == "band":
        return blk.get("aba_embed"), blk.get("slide_v3")
    return blk.get("wm_prob"), None


def evaluate_paired(
    clean_rows: dict[str, dict],
    wm_rows: dict[str, dict],
    method: str,
    attack_keys: list[str],
    *,
    suite: str,
    target_far: float,
) -> dict:
    """pos=已有 attack(wm)，neg=本脚本 attack(clean)。"""
    common = sorted(set(clean_rows) & set(wm_rows))
    out_rows = []
    if method == "band":
        for key in attack_keys:
            pos_a, neg_a, pos_s, neg_s = [], [], [], []
            for stem in common:
                cb = (clean_rows[stem].get("attacks") or {}).get(key) or {}
                wb = (wm_rows[stem].get("attacks") or {}).get(key) or {}
                if "error" in cb or "error" in wb:
                    continue
                wa, ws = _wm_scores_from_existing(wb, "band")
                ca, cs = cb.get("clean_aba"), cb.get("clean_slide")
                if None in (wa, ws, ca, cs):
                    continue
                pos_a.append(wa)
                neg_a.append(ca)
                pos_s.append(ws)
                neg_s.append(cs)
            pa, na = _finite(pos_a), _finite(neg_a)
            ps, ns = _finite(pos_s), _finite(neg_s)
            ha = roc_threshold_at_far(pa, na, target_far)
            hs = roc_threshold_at_far(ps, ns, target_far)
            t_a, t_s = ha.get("T"), hs.get("T")
            cas = None
            if t_a is not None and t_s is not None and pa.size and na.size:
                pos_c = cascade_score(pa, ps, t_a, t_s)
                neg_c = cascade_score(na, ns, t_a, t_s)
                cas = {
                    "tpr": rate_pack(int(np.sum(pos_c >= 1.0)), int(pos_c.size)),
                    "far": rate_pack(int(np.sum(neg_c >= 1.0)), int(neg_c.size)),
                    "auc": float(auc_mw(pos_c, neg_c)) if pos_c.size and neg_c.size else None,
                }
            out_rows.append(
                {
                    "attack": key,
                    "n_pos": int(pa.size),
                    "n_neg": int(na.size),
                    "aba": {
                        "T": t_a,
                        "far": rate_pack(int(np.sum(na >= t_a)), int(na.size)) if t_a is not None else None,
                        "tpr": rate_pack(int(np.sum(pa >= t_a)), int(pa.size)) if t_a is not None else None,
                        "auc": float(auc_mw(pa, na)) if pa.size and na.size else None,
                        "pos_mean": float(np.mean(pa)) if pa.size else None,
                        "neg_mean": float(np.mean(na)) if na.size else None,
                    },
                    "slide": {
                        "T": t_s,
                        "far": rate_pack(int(np.sum(ns >= t_s)), int(ns.size)) if t_s is not None else None,
                        "tpr": rate_pack(int(np.sum(ps >= t_s)), int(ps.size)) if t_s is not None else None,
                        "auc": float(auc_mw(ps, ns)) if ps.size and ns.size else None,
                        "pos_mean": float(np.mean(ps)) if ps.size else None,
                        "neg_mean": float(np.mean(ns)) if ns.size else None,
                    },
                    "cascade": cas,
                }
            )
    else:
        for key in attack_keys:
            pos, neg = [], []
            for stem in common:
                cb = (clean_rows[stem].get("attacks") or {}).get(key) or {}
                wb = (wm_rows[stem].get("attacks") or {}).get(key) or {}
                if "error" in cb or "error" in wb:
                    continue
                wp, _ = _wm_scores_from_existing(wb, method)
                cp = cb.get("clean_prob")
                if wp is None or cp is None:
                    continue
                pos.append(wp)
                neg.append(cp)
            pp, nn = _finite(pos), _finite(neg)
            hit = roc_threshold_at_far(pp, nn, target_far)
            T = hit.get("T")
            out_rows.append(
                {
                    "attack": key,
                    "n_pos": int(pp.size),
                    "n_neg": int(nn.size),
                    "T": T,
                    "far": rate_pack(int(np.sum(nn >= T)), int(nn.size)) if T is not None else None,
                    "tpr": rate_pack(int(np.sum(pp >= T)), int(pp.size)) if T is not None else None,
                    "auc": float(auc_mw(pp, nn)) if pp.size and nn.size else None,
                    "pos_mean": float(np.mean(pp)) if pp.size else None,
                    "neg_mean": float(np.mean(nn)) if nn.size else None,
                }
            )
    return {
        "attack_suite": suite,
        "method": method,
        "n_paired": len(common),
        "n_clean": len(clean_rows),
        "n_wm": len(wm_rows),
        "target_far": float(target_far),
        "protocol": "pos=existing attack(wm); neg=attack(clean)",
        "rows": out_rows,
    }


def _pct(pack) -> str:
    if pack is None:
        return "—"
    if isinstance(pack, dict):
        v = pack.get("value")
        return "—" if v is None else f"{100 * float(v):.2f}%"
    return f"{100 * float(pack):.2f}%"


def _num(x, nd=4) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def write_findings(out_root: Path, by_suite: dict) -> None:
    lines = [
        "# E2：攻击水印 vs 攻击干净",
        "",
        "- 正类：已有 `attack(wm)`（mild/anchor 三方法结果）",
        "- 负类：本脚本 `attack(clean)`",
        f"- 定标：每个攻击在负类 FAR≤{TARGET_FAR:.0%} 定 T",
        "",
    ]
    for suite, by_method in by_suite.items():
        keys = []
        for m in ("wavmark", "audioseal", "band"):
            for r in (by_method.get(m) or {}).get("rows") or []:
                if r["attack"] not in keys:
                    keys.append(r["attack"])
        lines += [f"## suite={suite}", "", "| attack | WavMark TPR/FAR | AudioSeal TPR/FAR | band 级联 TPR/FAR |", "|---|---|---|---|"]
        wm = {r["attack"]: r for r in (by_method.get("wavmark") or {}).get("rows") or []}
        as_ = {r["attack"]: r for r in (by_method.get("audioseal") or {}).get("rows") or []}
        bd = {r["attack"]: r for r in (by_method.get("band") or {}).get("rows") or []}
        for key in keys:
            w, a, b = wm.get(key, {}), as_.get(key, {}), bd.get(key, {})
            bc = b.get("cascade") or {}
            lines.append(
                f"| {key} | {_pct(w.get('tpr'))}/{_pct(w.get('far'))} | "
                f"{_pct(a.get('tpr'))}/{_pct(a.get('far'))} | "
                f"{_pct(bc.get('tpr'))}/{_pct(bc.get('far'))} |"
            )
        lines.append("")
        for method in ("wavmark", "audioseal", "band"):
            summ = by_method.get(method) or {}
            lines += [f"### {suite}/{method}", f"- paired={summ.get('n_paired')}", ""]
            if method == "band":
                lines += [
                    "| attack | aba TPR/FAR/AUC | slide TPR/FAR/AUC | cascade TPR/FAR/AUC |",
                    "|---|---|---|---|",
                ]
                for r in summ.get("rows") or []:
                    aa, ss, cc = r.get("aba") or {}, r.get("slide") or {}, r.get("cascade") or {}
                    lines.append(
                        f"| {r['attack']} | {_pct(aa.get('tpr'))}/{_pct(aa.get('far'))}/{_num(aa.get('auc'))} | "
                        f"{_pct(ss.get('tpr'))}/{_pct(ss.get('far'))}/{_num(ss.get('auc'))} | "
                        f"{_pct(cc.get('tpr'))}/{_pct(cc.get('far'))}/{_num(cc.get('auc'))} |"
                    )
            else:
                lines += ["| attack | T | TPR | FAR | AUC |", "|---|---:|---:|---:|---:|"]
                for r in summ.get("rows") or []:
                    lines.append(
                        f"| {r['attack']} | {_num(r.get('T'))} | {_pct(r.get('tpr'))} | "
                        f"{_pct(r.get('far'))} | {_num(r.get('auc'))} |"
                    )
            lines.append("")
    (out_root / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def run_one(args, suite: str, method: str, clips: list, out_suite: Path) -> None:
    out_m = out_suite / method
    for sub in ("per_clip", "errors", "logs"):
        (out_m / sub).mkdir(parents=True, exist_ok=True)
    attack_keys = suite_attack_keys(suite)
    common = {
        "method": method,
        "attack_suite": suite,
        "per_clip_dir": str(out_m / "per_clip"),
        "resume": bool(args.resume),
        "seed": int(args.seed),
        "cpu_threads": int(args.cpu_threads),
        "attack_keys": list(attack_keys),
        "moss_path": str(args.moss_path),
        "table": str(args.table),
        "as_gen": str(args.audioseal_generator),
        "as_det": str(args.audioseal_detector),
        "wm_ckpt": str(args.wavmark_checkpoint),
        "nbits": 16,
        "crop_keep": float(args.crop_keep),
        "snr_db": float(args.snr_db),
        "speed_factor": float(args.speed_factor),
    }
    n_workers = max(1, int(args.num_workers))
    print(
        f"=== suite={suite} method={method} CLEAN-ONLY workers={n_workers} "
        f"clips={len(clips)} attacks={len(attack_keys)} ===",
        flush=True,
    )
    payloads = build_payloads(
        clips,
        num_workers=n_workers,
        npu_ids=parse_npu_ids(args.npu_ids),
        device_single=str(args.device),
        logs_dir=out_m / "logs",
        common=common,
        items_key="items",
    )
    outs = run_shards(_worker, payloads, num_workers=n_workers)
    dump_json(out_m / "workers.json", outs)


def summarize_all(args, out_root: Path, suites: list[str], methods: list[str]) -> dict:
    by_suite = {}
    for suite in suites:
        attack_keys = suite_attack_keys(suite)
        wm_root = Path(args.wm_mild_root) if suite == "mild" else Path(args.wm_anchor_root)
        by_method = {}
        for method in methods:
            clean_rows = load_json_dir(out_root / suite / method / "per_clip")
            wm_rows = load_json_dir(wm_root / method / "per_clip") if wm_root.is_dir() else {}
            if not wm_rows:
                print(f"[warn] no wm rows for {suite}/{method} under {wm_root}", flush=True)
            summ = evaluate_paired(
                clean_rows,
                wm_rows,
                method,
                attack_keys,
                suite=suite,
                target_far=float(args.target_far),
            )
            dump_json(out_root / suite / method / "summary.json", summ)
            by_method[method] = summ
        by_suite[suite] = by_method
    dump_json(out_root / "all_summaries.json", by_suite)
    write_findings(out_root, by_suite)
    return by_suite


def main() -> None:
    ensure_spawn()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav-dir", type=str, default=default_mix_1500())
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--names", type=str, default="")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--suites", type=str, default="mild,anchor", help="mild,anchor")
    p.add_argument("--methods", type=str, default="audioseal,wavmark,band")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-far", type=float, default=TARGET_FAR)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--summary-only", action="store_true", help="只合并已有 clean + wm 结果")
    p.add_argument(
        "--wm-mild-root",
        type=str,
        default=str(EXP_ROOT / "mild_attack_three_n1500"),
        help="已有 mild 攻击水印结果根目录（含 {method}/per_clip）",
    )
    p.add_argument(
        "--wm-anchor-root",
        type=str,
        default=str(EXP_ROOT / "anchor_attack_three_n1500"),
        help="已有 anchor 攻击水印结果根目录",
    )
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--table", type=str, default=default_table_path())
    p.add_argument("--audioseal-generator", type=str, default=first_file(*AS_GEN_CANDIDATES))
    p.add_argument("--audioseal-detector", type=str, default=first_file(*AS_DET_CANDIDATES))
    p.add_argument("--wavmark-checkpoint", type=str, default=first_file(*WM_CKPT_CANDIDATES))
    p.add_argument("--crop-keep", type=float, default=0.7)
    p.add_argument("--snr-db", type=float, default=20.0)
    p.add_argument("--speed-factor", type=float, default=0.9)
    add_npu_args(p, default_workers=8, default_device="npu:0")
    args = p.parse_args()
    if args.no_resume:
        args.resume = False

    suites = [s.strip().lower() for s in args.suites.split(",") if s.strip()]
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if bool(args.summary_only):
        summarize_all(args, out_root, suites, methods)
        print(f"summary-only -> {out_root / 'FINDINGS.md'}", flush=True)
        return

    if str(args.manifest or "").strip():
        clips = load_clips_from_manifest(Path(args.manifest), limit=int(args.limit))
        src_desc = f"manifest={args.manifest}"
    else:
        name_set = {x.strip() for x in str(args.names).replace(" ", ",").split(",") if x.strip()} or None
        clips = list_clips(Path(args.wav_dir), limit=int(args.limit), names=name_set)
        src_desc = f"wav_dir={args.wav_dir}"
    if not clips:
        raise SystemExit(f"no clips ({src_desc})")

    (out_root / "RUN.md").write_text(
        "# E2 attack(clean) only\n\n"
        f"- source: `{src_desc}` n={len(clips)}\n"
        f"- suites: {suites}\n"
        f"- methods: {methods}\n"
        f"- only scores attack(clean); merge with wm-mild/wm-anchor roots later\n"
        f"- wm_mild_root: `{args.wm_mild_root}`\n"
        f"- wm_anchor_root: `{args.wm_anchor_root}`\n"
        f"- out: `{out_root}`\n",
        encoding="utf-8",
    )

    t0 = time.time()
    for suite in suites:
        for method in methods:
            run_one(args, suite, method, clips, out_root / suite)
    summarize_all(args, out_root, suites, methods)
    print(f"DONE {out_root} elapsed={((time.time() - t0) / 60):.1f} min", flush=True)
    print("FINDINGS:", out_root / "FINDINGS.md", flush=True)


if __name__ == "__main__":
    main()
