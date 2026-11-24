"""crop / speed 单路盲搜（argmax channel_max）· mix_1500 · 8×NPU。

**仅依赖本仓库 water_marks**（及 pip：librosa / soundfile / torch 等）。
不依赖 codec_an。

协议
----
- 不重定标；阈值写死本文件（random_50 mean_mid）：T_aba / T_slide
- 攻击：crop_mid keep=0.7；speed factor=0.9
- 盲搜：crop = 5ms head 步进；speed = 11 点 rate 网格；取 channel_max 最大
- 判决：channel_max ≥ T_aba ∨ channel_max ≥ T_slide（无 wave）
- 负类：干净无印（本脚本现场打分）

例::

  $env:PYTHONPATH = "e:\\code\\water_marks"
  python -m P1.experiments.run_blind_crop_speed_n1500 \\
    --wav-dir /path/to/wavs --out_dir /path/to/out \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7 --resume

  # 或用筛选清单
  python -m P1.experiments.run_blind_crop_speed_n1500 \\
    --manifest /path/to/manifest.jsonl --out_dir /path/to/out
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from base.attack import crop_mid, speed_wav
from base.channels import (
    DEFAULT_SPEED_FACTORS,
    P1_BAND_L9,
    profile_score_cfg,
    score_wav_once,
)
from base.config import (
    EXP_ROOT,
    default_cluster_dir,
    default_mix_1500,
    default_model_path,
    default_table_path,
)
from base.exp_parallel import flatten_key, launch_clip_workers, make_parser, resolve_out_dir
from base.moss_io import read_wav
from base.npu_parallel import ensure_spawn, free_device, limit_cpu_threads, open_worker_log, pin_npu
from base.rare_tokens import ensure_rare_table
from base.util import AUDIO_EXTS, dump_json, embed_wav, load_model
from base.wav_ms import FRAME_MS, align_ms_length, crop_wav_ms_range, ms_steps

OUT_STEM = "blind_crop_speed_n2000"
_LOCAL_EXP = Path(r"/data1/home/ynian/codes/eval/add_exp/result")
DEFAULT_OUT = (EXP_ROOT if EXP_ROOT.exists() else _LOCAL_EXP) / OUT_STEM

PERIOD = 3
STEP_MS = 5.0
# random_50 mean_mid（paper_band_L9）；判决 channel_max ≥ T_aba ∨ channel_max ≥ T_slide
T_ABA = 0.2943
T_SLIDE = 0.4543


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.4f}"


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_clips(
    wav_dir: Path,
    limit: int = 0,
    *,
    names: set[str] | None = None,
) -> list[tuple[str, str]]:
    """递归扫目录；可选 stem 白名单。返回 [(stem, path), ...]。"""
    files = [p for p in sorted(Path(wav_dir).rglob("*")) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if names:
        want = {n.strip() for n in names if str(n).strip()}
        files = [p for p in files if p.stem in want or p.name in want]
    if limit > 0:
        files = files[: int(limit)]
    return [(p.stem, str(p.resolve())) for p in files]


def load_clips_from_manifest(path: Path, limit: int = 0) -> list[tuple[str, str]]:
    """manifest.json / .jsonl，或每行一个 wav 路径。"""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"manifest not found: {path}")
    clips: list[tuple[str, str]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json" and text.startswith("{"):
        doc = json.loads(text)
        for it in doc.get("items") or []:
            p = Path(str(it.get("path") or ""))
            stem = str(it.get("stem") or (p.stem if p.name else ""))
            if stem and p.is_file():
                clips.append((stem, str(p.resolve())))
            elif stem and it.get("src") and Path(str(it["src"])).is_file():
                clips.append((stem, str(Path(it["src"]).resolve())))
    else:
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


def blind_ms_head(model, wav, sr, cfg, n_peel: int, period: int, step_ms: float) -> dict:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    dur_ms = float(wav.shape[0]) * 1000.0 / float(sr)
    period_ms = float(FRAME_MS) * int(period)
    best = None
    n_ok = 0
    for head in ms_steps(period_ms, step_ms):
        seg_ms = dur_ms - float(head)
        keep_ms = align_ms_length(seg_ms, period_ms)
        if keep_ms < period_ms:
            continue
        end_ms = float(head) + keep_ms
        sub = crop_wav_ms_range(wav, sr, head, end_ms)
        s = score_wav_once(model, sub, sr, cfg, n_peel=n_peel, period=period)
        n_ok += 1
        ch = float(s["channel_max"])
        rec = {
            "head_ms": float(head),
            "channel_max": ch,
            "aba_embed": float(s["aba_embed"]),
            "slide_v3": float(s["slide18"]),
        }
        if best is None or ch > best["channel_max"]:
            best = rec
    return {"n_trials": n_ok, "best": best}


def blind_speed(model, wav, sr, cfg, n_peel: int, period: int) -> dict:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    best = None
    n_ok = 0
    for fac in DEFAULT_SPEED_FACTORS:
        try:
            sub = speed_wav(wav, int(sr), float(fac))
        except Exception:
            continue
        if sub.size < 1000:
            continue
        s = score_wav_once(model, sub, sr, cfg, n_peel=n_peel, period=period)
        n_ok += 1
        ch = float(s["channel_max"])
        rec = {
            "factor": float(fac),
            "channel_max": ch,
            "aba_embed": float(s["aba_embed"]),
            "slide_v3": float(s["slide18"]),
        }
        if best is None or ch > best["channel_max"]:
            best = rec
    return {"n_trials": n_ok, "best": best}


def process_one(model, sr, table, name: str, wav_path: Path) -> dict:
    wav = read_wav(str(wav_path), sr)
    emb = embed_wav(
        model, wav, table, high_start=9, high_end=9, layers=[9], period=PERIOD
    )
    wm = np.asarray(emb["wav"], dtype=np.float32).reshape(-1)
    cfg = profile_score_cfg(table, P1_BAND_L9)
    n_peel = 9

    # 干净分数：现场打分（不再依赖 paper dump）
    so = score_wav_once(model, wav, sr, cfg, n_peel=n_peel, period=PERIOD)
    sw = score_wav_once(model, wm, sr, cfg, n_peel=n_peel, period=PERIOD)

    crop_w = crop_mid(wm, sr, keep=0.7)
    spd_w = speed_wav(wm, sr, factor=0.9)

    return {
        "name": name,
        "clean_orig_aba": float(so["aba_embed"]),
        "clean_orig_slide": float(so["slide18"]),
        "clean_wm_aba": float(sw["aba_embed"]),
        "clean_wm_slide": float(sw["slide18"]),
        "crop_mid": {
            "blind_wm": blind_ms_head(
                model, crop_w, sr, cfg, n_peel, PERIOD, STEP_MS
            ),
        },
        "speed": {
            "blind_wm": blind_speed(model, spd_w, sr, cfg, n_peel, PERIOD),
        },
    }


def decide_cascade(score_ch: float, t_aba: float, t_slide: float) -> bool:
    return float(score_ch) >= float(t_aba) or float(score_ch) >= float(t_slide)


def decide_cascade_pair(aba: float, slide: float, t_aba: float, t_slide: float) -> bool:
    return float(aba) >= float(t_aba) or float(slide) >= float(t_slide)


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


def evaluate(rows: list[dict], calib_names: set[str], t_aba: float, t_slide: float) -> dict:
    eval_rows = [r for r in rows if r["name"] not in calib_names]
    if len(eval_rows) < 10:
        eval_rows = rows

    def pack(pos, neg_scores_for_auc, hit_pos, hit_neg, n_neg):
        pos = np.asarray(pos, dtype=np.float64)
        neg = np.asarray(neg_scores_for_auc, dtype=np.float64)
        return {
            "n_pos": int(pos.size),
            "n_neg": int(n_neg),
            "mean_wm": float(np.mean(pos)) if pos.size else None,
            "mean_neg_score": float(np.mean(neg)) if neg.size else None,
            "auc": auc_mw(pos, neg) if neg.size and pos.size else None,
            "tpr": float(hit_pos / pos.size) if pos.size else None,
            "far": float(hit_neg / n_neg) if n_neg else None,
        }

    out = {
        "n_rows": len(rows),
        "n_eval": len(eval_rows),
        "T_aba": t_aba,
        "T_slide": t_slide,
        "by": {},
    }

    for ak in ("crop_mid", "speed"):
        pos, neg_auc = [], []
        hit_pos = hit_neg = 0
        for r in eval_rows:
            b = (r[ak]["blind_wm"] or {}).get("best")
            if not b:
                continue
            ch = float(b["channel_max"])
            pos.append(ch)
            neg_ch = max(float(r["clean_orig_aba"]), float(r["clean_orig_slide"]))
            neg_auc.append(neg_ch)
            if decide_cascade(ch, t_aba, t_slide):
                hit_pos += 1
            if decide_cascade_pair(r["clean_orig_aba"], r["clean_orig_slide"], t_aba, t_slide):
                hit_neg += 1
        out["by"][f"{ak}|blind|neg=clean"] = pack(pos, neg_auc, hit_pos, hit_neg, len(eval_rows))

    return out


def write_md(out_dir: Path, report: dict) -> None:
    lines = [
        "# crop / speed 盲搜（water_marks 自包含 · 8×NPU · mix_1500）",
        "",
        f"- 划分：random_50 定标（若有 split.json）；评估 hold-out"
        f"（n_eval={report['n_eval']} / n={report['n_rows']}）",
        f"- 阈值：T_aba={report['T_aba']:.4f}，T_slide={report['T_slide']:.4f}；"
        "判决 channel_max≥T_aba ∨ channel_max≥T_slide",
        "- 盲搜：单路 argmax channel_max（crop=5ms head；speed=11 rate）",
        "- 干净负类分数：本仓库现场 band 打分",
        "",
        "| 条件 | FAR↓ | TPR↑ | AUC↑ | μ_score(wm) | n_pos | n_neg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, m in (report.get("by") or {}).items():
        far = "—" if m["far"] is None else f"{100 * m['far']:.2f}%"
        tpr = "—" if m["tpr"] is None else f"{100 * m['tpr']:.2f}%"
        auc = "—" if m["auc"] is None else f"{m['auc']:.4f}"
        mu = "—" if m["mean_wm"] is None else f"{m['mean_wm']:.4f}"
        lines.append(
            f"| `{k}` | {far} | {tpr} | {auc} | {mu} | {m['n_pos']} | {m['n_neg']} |"
        )
    lines.append("")
    (out_dir / "COMPARE_BLIND.md").write_text("\n".join(lines), encoding="utf-8")


def _worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    pin_npu(str(payload["device"]))
    log_fp, log = open_worker_log(payload)

    model, sr = load_model(payload["model_path"], str(payload["device"]))
    table = ensure_rare_table(path=payload.get("table_path") or default_table_path())
    per_clip_dir = Path(payload["per_clip_dir"])
    per_clip_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(payload.get("resume"))

    rows = []
    n_err = 0
    for item in payload["items"]:
        # 支持 (stem, path) 或 纯 stem（兼容旧调用）
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            name, path_str = str(item[0]), str(item[1])
            wp = Path(path_str)
        else:
            name = str(item)
            wav_dir = Path(payload["wav_root"])
            wp = None
            for ext in (".wav", ".mp3", ".flac", ".ogg"):
                cand = wav_dir / f"{name}{ext}"
                if cand.is_file():
                    wp = cand
                    break
            if wp is None:
                hits = list(wav_dir.rglob(f"{name}.*"))
                wp = next((h for h in hits if h.suffix.lower() in AUDIO_EXTS), None)
        out_p = per_clip_dir / f"{name}.json"
        if resume and out_p.exists():
            try:
                rows.append(_load(out_p))
                log(f"resume {name}")
                continue
            except Exception:
                pass
        if wp is None or not Path(wp).is_file():
            n_err += 1
            log(f"FAIL {name}: missing wav")
            continue
        try:
            rec = process_one(model, sr, table, name, Path(wp))
            dump_json(out_p, rec)
            rows.append(rec)
            cb = ((rec.get("crop_mid") or {}).get("blind_wm") or {}).get("best") or {}
            sb = ((rec.get("speed") or {}).get("blind_wm") or {}).get("best") or {}
            log(
                f"ok {name} crop_ch={_fmt(cb.get('channel_max'))} "
                f"speed_ch={_fmt(sb.get('channel_max'))}"
            )
        except Exception as exc:
            n_err += 1
            log(f"FAIL {name}: {type(exc).__name__}: {exc}")

    del model
    free_device()
    if log_fp is not None:
        log_fp.close()
    return {
        "worker_id": int(payload["worker_id"]),
        "rows": rows,
        "n_ok": len(rows),
        "n_error": int(n_err),
    }


def main() -> None:
    p = make_parser(__doc__, default_out=str(DEFAULT_OUT), with_names=False)
    p.add_argument(
        "--manifest",
        type=str,
        default="",
        help="指定列表：manifest.json/.jsonl 或每行一个 wav 路径（优先于 --wav-dir）",
    )
    p.add_argument(
        "--names",
        type=str,
        default="",
        help="仅跑这些 stem（逗号分隔）；配合 --wav-dir",
    )
    p.add_argument(
        "--split-json",
        type=str,
        default="",
        help="可选：含 calib_names 的 split.json；缺省则全量作评估",
    )
    p.add_argument("--t-aba", type=float, default=T_ABA, help="覆盖默认 T_aba")
    p.add_argument("--t-slide", type=float, default=T_SLIDE, help="覆盖默认 T_slide")
    p.add_argument("--cluster-dir", type=str, default=default_cluster_dir())
    p.add_argument("--limit", type=int, default=0, help="0=全量")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if not str(args.manifest or "").strip():
        if not Path(args.wav_dir).exists():
            args.wav_dir = default_mix_1500()
    if not Path(str(args.model_path)).exists():
        args.model_path = default_model_path()
    if not Path(str(args.table)).exists():
        args.table = default_table_path()

    if str(args.manifest or "").strip():
        clips = load_clips_from_manifest(Path(args.manifest), limit=int(args.limit))
        src_desc = f"manifest={args.manifest}"
    else:
        name_set = {x.strip() for x in str(args.names).replace(" ", ",").split(",") if x.strip()} or None
        clips = list_clips(Path(args.wav_dir), limit=int(args.limit), names=name_set)
        src_desc = f"wav_dir={args.wav_dir}" + (f" names={len(name_set)}" if name_set else "")
    if not clips:
        raise SystemExit(f"no audio ({src_desc})")

    t_aba = float(args.t_aba)
    t_slide = float(args.t_slide)
    calib_names: set[str] = set()
    if str(args.split_json or "").strip() and Path(args.split_json).exists():
        calib_names = set((_load(args.split_json).get("calib_names") or []))

    out_dir = resolve_out_dir(args, OUT_STEM)
    per_clip_dir = out_dir / "per_clip"
    per_clip_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"blind crop/speed n={len(clips)} workers={args.num_workers} "
        f"T_aba={t_aba:.4f} T_slide={t_slide:.4f} src={src_desc} out={out_dir}",
        flush=True,
    )

    outs = launch_clip_workers(
        _worker,
        clips,
        args,
        out_dir=out_dir,
        common={
            "per_clip_dir": str(per_clip_dir),
            "cluster_dir": str(args.cluster_dir),
            "resume": bool(args.resume),
        },
    )
    rows = flatten_key(outs, "rows")
    if args.resume:
        disk = []
        for stem, _p in clips:
            fp = per_clip_dir / f"{stem}.json"
            if fp.exists():
                disk.append(_load(fp))
        if len(disk) >= len(rows):
            rows = disk

    report = evaluate(rows, calib_names, t_aba, t_slide)
    dump_json(out_dir / "blind_report.json", report)
    write_md(out_dir, report)
    dump_json(
        out_dir / "summary.json",
        {
            "experiment": OUT_STEM,
            "deps": "water_marks_only",
            "source": src_desc,
            "n_clips": len(clips),
            "n_ok": len(rows),
            "T_aba": t_aba,
            "T_slide": t_slide,
            "wav_dir": str(getattr(args, "wav_dir", "")),
            "manifest": str(args.manifest or ""),
            "out_dir": str(out_dir),
            "split_json": str(args.split_json or ""),
            "report": report,
        },
    )
    print(f"done n_ok={len(rows)} -> {out_dir / 'COMPARE_BLIND.md'}", flush=True)
    for k, m in (report.get("by") or {}).items():
        print(
            f"  {k}: FAR={_fmt(m.get('far'))} TPR={_fmt(m.get('tpr'))} AUC={_fmt(m.get('auc'))}",
            flush=True,
        )


if __name__ == "__main__":
    ensure_spawn()
    main()
