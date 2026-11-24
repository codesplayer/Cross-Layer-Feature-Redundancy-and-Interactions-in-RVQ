"""AnchorWM 十类严苛攻击 × {audioseal, wavmark, band_L9} × mix_1500 × 8×NPU。

**在 water_marks / P1 内完成**（不依赖 codec_an 源码）。
可选 pip：audioseal、wavmark；band 仅需本仓库 MOSS 栈。

协议
----
- 攻击：``base.anchor_attacks``（与 AnchorWM 论文 Distortions 对齐）
- 数据：mix_1500；三方法串行，各用满 ``--num-workers`` × NPU
- 判决：far01 冻结阈值（与 COMPARE_FAR01 一致，干净负类）
- 默认可落盘**干净条件**嵌印语音：``{out}/{method}/wav_wm_clean/{stem}_wm.wav``
- 可 ``--from-wm-dir`` 复用已嵌印 wav（只攻击+检测；仍会拷到 wav_wm_clean）

例::

  $env:PYTHONPATH = "E:\\code\\water_marks"
  python -m P1.experiments.run_anchor_attack_three_n1500 \\
    --methods audioseal,wavmark,band \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from base.anchor_attacks import DEFAULT_ANCHOR_KEYS, build_anchor_attacks
from base.channels import P1_BAND_L9, profile_score_cfg, score_wav_once
from base.config import EXP_ROOT, default_mix_1500, default_model_path, default_table_path, first_existing
from base.moss_io import read_wav, set_seed, write_wav
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
from base.util import AUDIO_EXTS, dump_json, embed_wav, load_model
from P1.baselines import (
    AS_DET_CANDIDATES,
    AS_GEN_CANDIDATES,
    WM_CKPT_CANDIDATES,
    audioseal_detect,
    audioseal_embed,
    disable_torch_compile_on_npu,
    first_file,
    load_audioseal,
    load_wavmark,
    wavmark_detect,
    wavmark_embed,
)

OUT_STEM = "anchor_attack_three_n2000"
_OUT_CANDS = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
    Path(r"E:\code\water_marks\result") / OUT_STEM,
    EXP_ROOT / OUT_STEM,
)
DEFAULT_OUT = next((p for p in _OUT_CANDS if p.parent.exists()), _OUT_CANDS[0])

# COMPARE_FAR01 冻结阈值
T_WM = 1e-12
T_AS = 0.18587467074494227
T_ABA = 0.08451842905452735
T_SLIDE = 0.3238710454914386

SPLIT_CANDS = (
    Path(r"D:\data\codec_an\anas\band\attack_layer_gap_n1500_calib\random_500\split.json"),
    Path(r"/data1/home/ynian/codes/codec_an/anas/band/attack_layer_gap_n1500_calib/random_500/split.json"),
)


def list_clips(
    wav_dir: Path,
    limit: int = 0,
    *,
    names: set[str] | None = None,
) -> list[tuple[str, str]]:
    files = [p for p in sorted(wav_dir.rglob("*")) if p.suffix.lower() in AUDIO_EXTS]
    if names:
        want = {n.strip() for n in names if str(n).strip()}
        files = [p for p in files if p.stem in want or p.name in want]
    if limit > 0:
        files = files[: int(limit)]
    return [(p.stem, str(p)) for p in files]


def load_clips_from_manifest(path: Path, limit: int = 0) -> list[tuple[str, str]]:
    """读筛选脚本写出的 manifest.json / .jsonl，或每行一个 wav 路径的列表。"""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"manifest not found: {path}")
    clips: list[tuple[str, str]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # JSON 汇总
    if path.suffix.lower() == ".json" and text.startswith("{"):
        doc = json.loads(text)
        items = doc.get("items") or []
        for it in items:
            p = Path(str(it.get("path") or ""))
            stem = str(it.get("stem") or (p.stem if p.name else ""))
            if stem and p.is_file():
                clips.append((stem, str(p.resolve())))
            elif stem and it.get("src") and Path(str(it["src"])).is_file():
                clips.append((stem, str(Path(it["src"]).resolve())))
    else:
        # jsonl 或纯路径列表
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                it = json.loads(line)
                p = Path(str(it.get("path") or it.get("src") or ""))
                stem = str(it.get("stem") or (p.stem if p.name else ""))
                if stem and p.is_file():
                    clips.append((stem, str(p.resolve())))
            else:
                p = Path(line)
                if p.is_file():
                    clips.append((p.stem, str(p.resolve())))
    # 去重保序
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for stem, p in clips:
        if stem in seen:
            continue
        seen.add(stem)
        uniq.append((stem, p))
    if limit > 0:
        uniq = uniq[: int(limit)]
    return uniq


def find_wm_wav(stem: str, wm_dir: Path | None) -> Path | None:
    if wm_dir is None or not wm_dir.is_dir():
        return None
    for name in (f"{stem}_wm.wav", f"{stem}.wav", f"{stem}_wm.flac"):
        p = wm_dir / name
        if p.is_file():
            return p
    hits = list(wm_dir.glob(f"{stem}*"))
    return hits[0] if hits else None


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_clean_wm_wav(
    clean_wm_dir: Path | None,
    stem: str,
    wm: np.ndarray,
    sr: int,
    *,
    from_wm: Path | None = None,
) -> str | None:
    """保存干净嵌印语音到 ``{clean_wm_dir}/{stem}_wm.wav``；返回路径或 None。"""
    if clean_wm_dir is None:
        return None
    clean_wm_dir.mkdir(parents=True, exist_ok=True)
    dest = clean_wm_dir / f"{stem}_wm.wav"
    if from_wm is not None and Path(from_wm).is_file():
        # 复用已有 wm：同格式直接拷，否则重写为 float wav
        src = Path(from_wm)
        if src.resolve() == dest.resolve():
            return str(dest)
        if src.suffix.lower() == ".wav":
            shutil.copy2(src, dest)
            return str(dest)
    write_wav(dest, np.asarray(wm, dtype=np.float32).reshape(-1), int(sr))
    return str(dest)


def process_band(
    model, sr, table, stem, wav_path, attacks, from_wm: Path | None, clean_wm_dir: Path | None
) -> dict:
    wav = np.asarray(read_wav(str(wav_path), sr), dtype=np.float32).reshape(-1)
    if from_wm is not None:
        wm = np.asarray(read_wav(str(from_wm), sr), dtype=np.float32).reshape(-1)
        n = min(wav.shape[0], wm.shape[0])
        wav, wm = wav[:n], wm[:n]
    else:
        emb = embed_wav(model, wav, table, high_start=9, high_end=9, layers=[9], period=3)
        wm = np.asarray(emb["wav"], dtype=np.float32).reshape(-1)
    wm_path = save_clean_wm_wav(clean_wm_dir, stem, wm, int(sr), from_wm=from_wm)
    cfg = profile_score_cfg(table, P1_BAND_L9)
    so = score_wav_once(model, wav, sr, cfg, n_peel=9, period=3)
    sw = score_wav_once(model, wm, sr, cfg, n_peel=9, period=3)
    atk = {}
    for key, fn, meta in attacks:
        try:
            w2 = fn(wm, int(sr))
            s = score_wav_once(model, w2, sr, cfg, n_peel=9, period=3)
            atk[key] = {
                "aba_embed": float(s["aba_embed"]),
                "slide_v3": float(s["slide18"]),
                "channel_max": float(s["channel_max"]),
                "meta": meta,
            }
        except Exception as e:
            atk[key] = {"error": f"{type(e).__name__}: {e}", "meta": meta}
    rec = {
        "name": stem,
        "method": "band",
        "clean_orig_aba": float(so["aba_embed"]),
        "clean_orig_slide": float(so["slide18"]),
        "clean_wm_aba": float(sw["aba_embed"]),
        "clean_wm_slide": float(sw["slide18"]),
        "attacks": atk,
        "sr": int(sr),
    }
    if wm_path:
        rec["clean_wm_wav"] = wm_path
    return rec


def process_audioseal(
    gen, det, device, stem, wav_path, sr_src, attacks, from_wm: Path | None, clean_wm_dir: Path | None
) -> dict:
    wav16 = np.asarray(read_wav(str(wav_path), 16000), dtype=np.float32).reshape(-1)
    if from_wm is not None:
        wm16 = np.asarray(read_wav(str(from_wm), 16000), dtype=np.float32).reshape(-1)
    else:
        wm16, _msg = audioseal_embed(gen, wav16, 16000, device)
    wm_path = save_clean_wm_wav(clean_wm_dir, stem, wm16, 16000, from_wm=from_wm)
    p_orig = audioseal_detect(det, wav16, 16000, device)
    p_wm = audioseal_detect(det, wm16, 16000, device)
    atk = {}
    for key, fn, meta in attacks:
        try:
            w2 = fn(wm16, 16000)
            atk[key] = {"wm_prob": audioseal_detect(det, w2, 16000, device), "meta": meta}
        except Exception as e:
            atk[key] = {"error": f"{type(e).__name__}: {e}", "meta": meta}
    rec = {
        "name": stem,
        "method": "audioseal",
        "clean_orig_prob": float(p_orig),
        "clean_wm_prob": float(p_wm),
        "attacks": atk,
        "sr": 16000,
    }
    if wm_path:
        rec["clean_wm_wav"] = wm_path
    return rec


def process_wavmark(
    model, stem, wav_path, attacks, from_wm: Path | None, clean_wm_dir: Path | None
) -> dict:
    import librosa

    wav_n, sr0 = librosa.load(str(wav_path), sr=None, mono=True)
    wav = np.asarray(wav_n, dtype=np.float32).reshape(-1)
    sr0 = int(sr0)
    payload = None
    if from_wm is not None:
        wm = np.asarray(librosa.load(str(from_wm), sr=sr0, mono=True)[0], dtype=np.float32).reshape(-1)
    else:
        wm, payload, _ = wavmark_embed(model, wav, sr0)
    wm_path = save_clean_wm_wav(clean_wm_dir, stem, wm, sr0, from_wm=from_wm)
    d_o = wavmark_detect(model, wav, sr0, payload)
    d_w = wavmark_detect(model, wm, sr0, payload)
    atk = {}
    for key, fn, meta in attacks:
        try:
            w2 = fn(wm, sr0)
            d = wavmark_detect(model, w2, sr0, payload)
            atk[key] = {"wm_prob": float(d["wm_prob"]), "bit_acc": d.get("bit_acc"), "meta": meta}
        except Exception as e:
            atk[key] = {"error": f"{type(e).__name__}: {e}", "meta": meta}
    rec = {
        "name": stem,
        "method": "wavmark",
        "clean_orig_prob": float(d_o["wm_prob"]),
        "clean_wm_prob": float(d_w["wm_prob"]),
        "attacks": atk,
        "sr": int(sr0),
    }
    if wm_path:
        rec["clean_wm_wav"] = wm_path
    return rec


def _worker(payload: dict) -> dict:
    log_fp, log = open_worker_log(payload)
    wid = int(payload.get("worker_id", 0))
    method = str(payload["method"])
    device = str(payload["device"])
    out_pc = Path(payload["per_clip_dir"])
    resume = bool(payload.get("resume", True))
    attacks = build_anchor_attacks(payload.get("attack_keys"))
    n_ok = n_err = 0
    wm_dir = Path(payload["wm_dir"]) if payload.get("wm_dir") else None
    clean_wm_dir = Path(payload["clean_wm_dir"]) if payload.get("clean_wm_dir") else None
    try:
        limit_cpu_threads(int(payload.get("cpu_threads", 1) or 1))
        pin_npu(device)
        set_seed(int(payload.get("seed", 0)) + wid)
        log(
            f"worker{wid} method={method} device={device} n={len(payload['items'])} "
            f"save_clean_wm={bool(clean_wm_dir)}"
        )

        band_model = table = sr = None
        as_gen = as_det = None
        wm_model = None
        if method == "band":
            band_model, sr = load_model(str(payload["moss_path"]), device)
            table = ensure_rare_table(str(payload["table"]))
        elif method == "audioseal":
            disable_torch_compile_on_npu(device)
            as_gen, as_det = load_audioseal(
                str(payload["as_gen"]), str(payload["as_det"]), device, nbits=int(payload.get("nbits", 16))
            )
        elif method == "wavmark":
            wm_model = load_wavmark(str(payload["wm_ckpt"]), device)
        else:
            raise SystemExit(f"unknown method {method}")

        for i, (stem, path) in enumerate(payload["items"]):
            dest = out_pc / f"{stem}.json"
            clean_path = (clean_wm_dir / f"{stem}_wm.wav") if clean_wm_dir else None
            # resume：json 在且（不要求落盘 或 干净 wm 已在）才跳过
            if resume and dest.is_file() and (clean_path is None or clean_path.is_file()):
                n_ok += 1
                continue
            try:
                from_wm = find_wm_wav(stem, wm_dir)
                if method == "band":
                    rec = process_band(
                        band_model, sr, table, stem, path, attacks, from_wm, clean_wm_dir
                    )
                elif method == "audioseal":
                    rec = process_audioseal(
                        as_gen, as_det, device, stem, path, 16000, attacks, from_wm, clean_wm_dir
                    )
                else:
                    rec = process_wavmark(wm_model, stem, path, attacks, from_wm, clean_wm_dir)
                _write_json(dest, rec)
                n_ok += 1
            except Exception as e:
                n_err += 1
                log(f"ERR {stem}: {e}\n{traceback.format_exc()}")
                _write_json(out_pc.parent / "errors" / f"{stem}.json", {"name": stem, "error": str(e)})
            if (i + 1) % 20 == 0 or i == 0:
                log(f"progress {i+1}/{len(payload['items'])} ok={n_ok} err={n_err}")
        return {"worker_id": wid, "method": method, "n_ok": n_ok, "n_error": n_err}
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        return {"worker_id": wid, "method": method, "n_ok": n_ok, "n_error": n_err + 1, "error": str(e)}
    finally:
        free_device()
        if log_fp:
            log_fp.close()


def load_split(path: str) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.is_file():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")).get("calib_names") or [])


def merge_method(out_method: Path, method: str, calib: set[str]) -> dict:
    files = sorted((out_method / "per_clip").glob("*.json"))
    hold = []
    for fp in files:
        rec = json.loads(fp.read_text(encoding="utf-8"))
        if calib and rec.get("name") in calib:
            continue
        hold.append(rec)

    def rate(xs, T):
        xs = [x for x in xs if x is not None and np.isfinite(float(x))]
        return float(np.mean(np.asarray(xs, float) >= T)) if xs else None

    def mean(xs):
        xs = [x for x in xs if x is not None and np.isfinite(float(x))]
        return float(np.mean(xs)) if xs else None

    rows = []
    if method == "band":
        neg_a = [r["clean_orig_aba"] for r in hold]
        neg_s = [r["clean_orig_slide"] for r in hold]
        for key in DEFAULT_ANCHOR_KEYS:
            pos_a, pos_s = [], []
            for r in hold:
                blk = (r.get("attacks") or {}).get(key) or {}
                if "error" in blk:
                    continue
                pos_a.append(blk.get("aba_embed"))
                pos_s.append(blk.get("slide_v3"))
            cas_far = float(
                np.mean([(na >= T_ABA or ns >= T_SLIDE) for na, ns in zip(neg_a, neg_s)])
            ) if hold else None
            cas_tpr = float(
                np.mean([(pa >= T_ABA or ps >= T_SLIDE) for pa, ps in zip(pos_a, pos_s)])
            ) if pos_a else None
            rows.append(
                {
                    "attack": key,
                    "mean_wm_aba": mean(pos_a),
                    "far_aba": rate(neg_a, T_ABA),
                    "tpr_aba": rate(pos_a, T_ABA),
                    "cascade_far": cas_far,
                    "cascade_tpr": cas_tpr,
                    "n": len(pos_a),
                }
            )
    else:
        T = T_AS if method == "audioseal" else T_WM
        neg = [r["clean_orig_prob"] for r in hold]
        for key in DEFAULT_ANCHOR_KEYS:
            pos = []
            for r in hold:
                blk = (r.get("attacks") or {}).get(key) or {}
                if "error" in blk:
                    continue
                pos.append(blk.get("wm_prob"))
            rows.append(
                {
                    "attack": key,
                    "mean_wm": mean(pos),
                    "mean_orig": mean(neg),
                    "far": rate(neg, T),
                    "tpr": rate(pos, T),
                    "T": T,
                    "n": len(pos),
                }
            )
    summary = {
        "method": method,
        "n_files": len(files),
        "n_eval": len(hold),
        "n_calib_removed": len(files) - len(hold),
        "protocol": "far01_frozen",
        "rows": rows,
    }
    dump_json(out_method / "summary_far01.json", summary)
    return summary


def write_compare_md(out_root: Path, by_method: dict) -> None:
    lines = [
        "# Anchor 十类攻击 @ far01（P1 三方法）",
        "",
        "阈值冻结自 COMPARE_FAR01；评估 hold-out（去掉 random_500 定标集，若提供 split）。",
        "",
        "| attack | WavMark TPR/FAR | AudioSeal TPR/FAR | band 级联 TPR/FAR |",
        "|---|---:|---:|---:|",
    ]

    def pct(x):
        return "—" if x is None else f"{100*float(x):.2f}%"

    wm = {r["attack"]: r for r in (by_method.get("wavmark") or {}).get("rows") or []}
    as_ = {r["attack"]: r for r in (by_method.get("audioseal") or {}).get("rows") or []}
    bd = {r["attack"]: r for r in (by_method.get("band") or {}).get("rows") or []}
    for key in DEFAULT_ANCHOR_KEYS:
        w, a, b = wm.get(key, {}), as_.get(key, {}), bd.get(key, {})
        lines.append(
            f"| {key} | {pct(w.get('tpr'))}/{pct(w.get('far'))} | "
            f"{pct(a.get('tpr'))}/{pct(a.get('far'))} | "
            f"{pct(b.get('cascade_tpr'))}/{pct(b.get('cascade_far'))} |"
        )
    (out_root / "COMPARE_ANCHOR_FAR01.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_method(args, method: str, clips: list[tuple[str, str]], out_root: Path) -> dict:
    out_m = out_root / method
    for sub in ("per_clip", "errors", "logs"):
        (out_m / sub).mkdir(parents=True, exist_ok=True)
    clean_wm_dir = None
    if bool(getattr(args, "save_clean_wm", True)):
        custom = str(getattr(args, "clean_wm_dir", "") or "").strip()
        # 自定义根目录时仍按 method 分子目录，避免三方法同 stem 互相覆盖
        clean_wm_dir = (Path(custom) / method) if custom else (out_m / "wav_wm_clean")
        clean_wm_dir.mkdir(parents=True, exist_ok=True)
    wm_map = {
        "audioseal": args.audioseal_wm_dir,
        "wavmark": args.wavmark_wm_dir,
        "band": args.band_wm_dir,
    }
    common = {
        "method": method,
        "per_clip_dir": str(out_m / "per_clip"),
        "resume": bool(args.resume),
        "seed": int(args.seed),
        "cpu_threads": int(args.cpu_threads),
        "attack_keys": list(DEFAULT_ANCHOR_KEYS),
        "wm_dir": str(wm_map.get(method) or "") or None,
        "clean_wm_dir": str(clean_wm_dir) if clean_wm_dir else None,
        "moss_path": str(args.moss_path),
        "table": str(args.table),
        "as_gen": str(args.audioseal_generator),
        "as_det": str(args.audioseal_detector),
        "wm_ckpt": str(args.wavmark_checkpoint),
        "nbits": 16,
    }
    # clear None wm_dir
    if not common["wm_dir"]:
        common["wm_dir"] = None
    n_workers = max(1, int(args.num_workers))
    print(
        f"=== {method} workers={n_workers} clips={len(clips)} "
        f"clean_wm={clean_wm_dir or 'off'} ===",
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
    calib = load_split(str(args.split_json or ""))
    return merge_method(out_m, method, calib)


def main():
    ensure_spawn()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--wav-dir",
        type=str,
        default=default_mix_1500(),
        help="语音目录（递归扫音频）；与 --manifest 二选一，manifest 优先",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default="",
        help="指定列表：fleurs 的 manifest.json/.jsonl，或每行一个 wav 路径",
    )
    p.add_argument(
        "--names",
        type=str,
        default="",
        help="仅跑这些 stem（逗号分隔）；配合 --wav-dir 过滤",
    )
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--methods", type=str, default="audioseal,wavmark,band")
    p.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 条")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--split-json", type=str, default=str(first_existing(*SPLIT_CANDS, "") or ""))
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--table", type=str, default=default_table_path())
    p.add_argument("--audioseal-generator", type=str, default=first_file(*AS_GEN_CANDIDATES))
    p.add_argument("--audioseal-detector", type=str, default=first_file(*AS_DET_CANDIDATES))
    p.add_argument("--wavmark-checkpoint", type=str, default=first_file(*WM_CKPT_CANDIDATES))
    p.add_argument("--audioseal-wm-dir", type=str, default="", help="已有 AS wav_wm 则只攻击+检测")
    p.add_argument("--wavmark-wm-dir", type=str, default="", help="已有 WM wav_wm")
    p.add_argument("--band-wm-dir", type=str, default="", help="已有 band wav_wm")
    p.add_argument(
        "--save-clean-wm",
        action="store_true",
        default=True,
        help="保存干净嵌印语音（默认开）→ {out}/{method}/wav_wm_clean/{stem}_wm.wav",
    )
    p.add_argument("--no-save-clean-wm", action="store_true", help="不落盘干净水印 wav")
    p.add_argument(
        "--clean-wm-dir",
        type=str,
        default="",
        help="干净水印输出目录；空则用 {out}/{method}/wav_wm_clean",
    )
    add_npu_args(p, default_workers=8, default_device="npu:0")
    args = p.parse_args()
    if args.no_resume:
        args.resume = False
    if args.no_save_clean_wm:
        args.save_clean_wm = False

    if str(args.manifest or "").strip():
        clips = load_clips_from_manifest(Path(args.manifest), limit=int(args.limit))
        src_desc = f"manifest={args.manifest}"
    else:
        name_set = {x.strip() for x in str(args.names).replace(" ", ",").split(",") if x.strip()} or None
        clips = list_clips(Path(args.wav_dir), limit=int(args.limit), names=name_set)
        src_desc = f"wav_dir={args.wav_dir}" + (f" names={len(name_set)}" if name_set else "")
    if not clips:
        raise SystemExit(f"no clips ({src_desc})")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    (out_root / "RUN.md").write_text(
        "# Anchor attack three methods (P1)\n\n"
        f"- source: `{src_desc}`\n"
        f"- n_clips: {len(clips)}\n"
        f"- methods: {methods}\n"
        f"- attacks: {list(DEFAULT_ANCHOR_KEYS)}\n"
        f"- save_clean_wm: {bool(args.save_clean_wm)}  "
        f"dir: `{args.clean_wm_dir or '{out}/{method}/wav_wm_clean'}`\n"
        f"- workers: {args.num_workers}  npu_ids: {args.npu_ids}\n"
        f"- split_json: `{args.split_json}`\n"
        "- deps: water_marks only (+ optional pip audioseal/wavmark)\n",
        encoding="utf-8",
    )

    summaries = {}
    t0 = time.time()
    for m in methods:
        summaries[m] = run_one_method(args, m, clips, out_root)
    write_compare_md(out_root, summaries)
    dump_json(out_root / "all_summaries.json", summaries)
    print(f"DONE {out_root} elapsed={((time.time()-t0)/60):.1f} min", flush=True)
    print("compare:", out_root / "COMPARE_ANCHOR_FAR01.md", flush=True)


if __name__ == "__main__":
    main()
