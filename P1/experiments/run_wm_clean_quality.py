"""嵌印干净语音 vs 原始语音：LSD / MCD / 说话人 / 情感 / DNSMOS。

重构根目录约定（``--recon-root``）::

  {recon-root}/{method}/wav_wm_clean/{stem}_wm.wav
  method ∈ {audioseal, band, wavmark}（可用 ``--methods`` 改）

原始语音用 ``--orig-dir``（或 ``--manifest``）按 stem 配对；
recon 侧常见 ``{stem}_wm.wav``，也接受 ``{stem}.wav``。

例::

  python -m P1.experiments.run_wm_clean_quality \\
    --orig-dir /data1/.../fleurs/select_n2000 \\
    --recon-root /data1/.../fleurs/wm_n2000 \\
    --out-dir /data1/.../result/wm_clean_quality_n2000 \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from base.config import EXP_ROOT, WAVLM_SV_PATH, first_existing
from base.io import match_len, write_wav_float
from base.lsd import CalcuLSD
from base.mcd import CalcuMCD
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
from base.util import dump_json
from P1.experiments.run_anchor_attack_three_n1500 import (
    find_wm_wav,
    list_clips,
    load_clips_from_manifest,
)

OUT_STEM = "wm_clean_quality"
_OUT_CANDS = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
    Path(r"E:\code\water_marks\result") / OUT_STEM,
    EXP_ROOT / OUT_STEM,
)
DEFAULT_OUT = next((p for p in _OUT_CANDS if p.parent.exists()), _OUT_CANDS[0])
DEFAULT_METHODS = ("audioseal", "band", "wavmark")
DEFAULT_WAV_SUBDIR = "wav_wm_clean"

METRIC_VERSION = 1
_EVAL_FILES = (
    Path("eval") / "experiments" / "calcu_speaker_sim.py",
    Path("eval") / "experiments" / "calcu_emotion_sim.py",
    Path("eval") / "experiments" / "calcu_dnsmos.py",
)
_WAVLM = (
    Path(r"/data1/home/ynian/pred_models/wavlm-base-plus-sv"),
    Path(r"/home/ma-user/work/ynian/code/codec_an/model/wavlm-base-plus-sv"),
    Path(r"E:\code\codec_an\model\wavlm-base-plus-sv"),
    Path(WAVLM_SV_PATH),
)
_EMO = (
    Path(r"/data1/home/ynian/pred_models/emotion2vec_plus_large"),
    Path(r"/home/ma-user/work/ynian/pred_model/emotion2vec_plus_large"),
    Path(r"E:\code\codec_an\model\emotion2vec_plus_large"),
)
_DNSMOS_DIRS = (
    Path(r"/data1/home/ynian/pred_models/DNSMOS"),
    Path(r"/home/ma-user/work/ynian/code/codec_an/model/DNSMOS"),
    Path(r"E:\code\codec_an\model\DNSMOS"),
    Path(r"D:\data\codec_an\model\DNSMOS"),
)

METRIC_KEYS = (
    "lsd",
    "mcd",
    "speaker_cos",
    "emotion_cos",
    "dnsmos_ovrl",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_p808",
)


def find_eval_root(explicit: str = "") -> Path:
    here = Path(__file__).resolve()
    candidates = [
        explicit,
        os.environ.get("CODEC_AN", ""),
        r"/data1/home/ynian/codes/eval/add_exp/water_marks",
        r"/data1/home/ynian/codes/codec_an",
        str(Path(r"E:\code\water_marks")),
        str(Path(r"E:\code\codec_an")),
    ]
    for parent in here.parents:
        candidates.append(str(parent / "codec_an"))
        candidates.append(str(parent))
    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        root = str(Path(raw))
        if root in seen:
            continue
        seen.add(root)
        if all((Path(root) / rel).is_file() for rel in _EVAL_FILES):
            return Path(root)
    raise SystemExit("cannot find eval (speaker/emotion/dnsmos); set --codec-an")


def resolve_dnsmos_dir(explicit: str = "") -> Path:
    cands = [Path(explicit)] if explicit else []
    cands.extend(_DNSMOS_DIRS)
    for d in cands:
        if d and (d / "sig_bak_ovr.onnx").is_file() and (d / "model_v8.onnx").is_file():
            return d
    raise SystemExit("cannot find DNSMOS onnx; set --dnsmos-dir")


def use_eval_root(root: Path) -> None:
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)


def unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
    except Exception:
        pass


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def clip_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        isinstance(doc, dict)
        and doc.get("metric_version") == METRIC_VERSION
        and doc.get("lsd") is not None
        and "error" not in doc
    )


def score_quality(
    *,
    ref_wav: np.ndarray,
    deg_wav: np.ndarray,
    sr: int,
    ref_path: str,
    tmp: Path,
    stem: str,
    lsd_tool: CalcuLSD,
    mcd_tool: CalcuMCD,
    spk,
    emo,
    dnsmos,
) -> dict:
    ref = np.asarray(ref_wav, dtype=np.float32).reshape(-1)
    deg = np.asarray(match_len(ref, deg_wav), dtype=np.float32).reshape(-1)
    out: dict = {}
    try:
        val, _ = lsd_tool.compute(ref, deg)
        out["lsd"] = float(val)
    except Exception as exc:
        out["lsd"] = None
        out["lsd_error"] = str(exc)
    rp = tmp / f"{stem}_ref.wav"
    dp = tmp / f"{stem}_deg.wav"
    write_wav_float(rp, ref, sr)
    write_wav_float(dp, deg, sr)
    try:
        out["mcd"] = float(mcd_tool.mcd_file(str(rp), str(dp)))
    except Exception as exc:
        out["mcd"] = None
        out["mcd_error"] = str(exc)
    try:
        mos = dnsmos(str(dp), 16000, False)
        out["speaker_cos"] = float(spk.sim_file(str(ref_path), str(dp)))
        out["emotion_cos"] = float(emo.sim_file(str(ref_path), str(dp)))
        out["dnsmos_ovrl"] = float(mos["OVRL"])
        out["dnsmos_sig"] = float(mos["SIG"])
        out["dnsmos_bak"] = float(mos["BAK"])
        out["dnsmos_p808"] = float(mos["P808_MOS"])
    except Exception as exc:
        out["attr_error"] = str(exc)
    finally:
        unlink_quiet(rp)
        unlink_quiet(dp)
    return out


def pair_clips(
    orig_clips: list[tuple[str, str]],
    recon_wav_dir: Path,
) -> list[tuple[str, str, str]]:
    """(stem, orig_path, recon_path)；缺 recon 的跳过。"""
    paired: list[tuple[str, str, str]] = []
    for stem, src in orig_clips:
        hit = find_wm_wav(stem, recon_wav_dir)
        if hit is None:
            continue
        paired.append((stem, src, str(hit.resolve())))
    return paired


def discover_methods(recon_root: Path, wav_subdir: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return list(explicit)
    found = []
    for name in DEFAULT_METHODS:
        if (recon_root / name / wav_subdir).is_dir():
            found.append(name)
    if found:
        return found
    # 兜底：扫一级子目录
    for p in sorted(recon_root.iterdir()):
        if p.is_dir() and (p / wav_subdir).is_dir():
            found.append(p.name)
    return found


def _worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    device = str(payload["device"])
    pin_npu(device)
    log_fp, log = open_worker_log(payload)
    per_dir = Path(payload["per_clip_dir"])
    tmp = Path(payload["tmp_dir"]) / f"w{payload['worker_id']}"
    per_dir.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    resume = bool(payload.get("resume"))
    sr = int(payload.get("sr") or 16000)
    method = str(payload.get("method") or "")
    n_ok = n_skip = n_err = 0

    use_eval_root(Path(payload["codec_an"]))
    from eval.experiments.calcu_dnsmos import ComputeScore
    from eval.experiments.calcu_emotion_sim import CalcuEmotionSim
    from eval.experiments.calcu_speaker_sim import CalcuSpeakerSim

    try:
        set_seed(int(payload.get("seed") or 0) + int(payload["worker_id"]))
        lsd_tool = CalcuLSD(sr=sr)
        mcd_tool = CalcuMCD()
        spk = CalcuSpeakerSim(model_source=str(payload["wavlm"]), device=device)
        emo = CalcuEmotionSim(model_id=str(payload["emotion2vec"]), device=device)
        dnsmos_dir = Path(payload["dnsmos_dir"])
        dnsmos = ComputeScore(
            str(dnsmos_dir / "sig_bak_ovr.onnx"),
            str(dnsmos_dir / "model_v8.onnx"),
            num_threads=1,
        )
        log(f"start method={method} clips={len(payload['items'])} sr={sr}")

        for i, item in enumerate(payload["items"]):
            stem, orig_path, recon_path = item[0], item[1], item[2]
            safe = str(stem).replace("/", "_").replace("\\", "_")
            out_json = per_dir / f"{safe}.json"
            if resume and clip_ok(out_json):
                n_skip += 1
                n_ok += 1
                continue
            try:
                ref = np.asarray(read_wav(orig_path, sr), dtype=np.float32).reshape(-1)
                deg = np.asarray(read_wav(recon_path, sr), dtype=np.float32).reshape(-1)
                qual = score_quality(
                    ref_wav=ref,
                    deg_wav=deg,
                    sr=sr,
                    ref_path=orig_path,
                    tmp=tmp,
                    stem=safe,
                    lsd_tool=lsd_tool,
                    mcd_tool=mcd_tool,
                    spk=spk,
                    emo=emo,
                    dnsmos=dnsmos,
                )
                _write_json(
                    out_json,
                    {
                        "stem": stem,
                        "method": method,
                        "src": orig_path,
                        "recon_wav": recon_path,
                        "sr": sr,
                        "metric_version": METRIC_VERSION,
                        "n_samples": int(ref.shape[0]),
                        **qual,
                    },
                )
                n_ok += 1
            except Exception as exc:
                n_err += 1
                _write_json(
                    out_json,
                    {
                        "stem": stem,
                        "method": method,
                        "src": orig_path,
                        "recon_wav": recon_path,
                        "error": str(exc),
                        "trace": traceback.format_exc(),
                    },
                )
                log(f"ERR {stem}: {exc}")
            if (i + 1) % 20 == 0:
                log(f"progress {i+1}/{len(payload['items'])} ok={n_ok} skip={n_skip} err={n_err}")
    finally:
        free_device()
        if log_fp:
            log_fp.close()
    return {
        "worker_id": payload["worker_id"],
        "method": method,
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_err": n_err,
    }


def _mean(xs) -> float | None:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return None if not vals else float(np.mean(vals))


def aggregate(per_dir: Path, *, method: str = "") -> dict:
    docs = []
    for p in sorted(per_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("error") or d.get("metric_version") != METRIC_VERSION:
            continue
        docs.append(d)
    return {
        "method": method,
        "n_clips": len(docs),
        "metric_version": METRIC_VERSION,
        "mean": {k: _mean([d.get(k) for d in docs]) for k in METRIC_KEYS},
    }


def render_md(summaries: list[dict], *, title: str = "wm_clean quality") -> str:
    def f(x):
        return "-" if x is None else f"{float(x):.4f}"

    lines = [
        f"# {title}\n",
        f"metric_v={METRIC_VERSION}\n",
        "| method | n | LSD | MCD | speaker | emotion | DNSMOS_ovrl | DNSMOS_p808 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        m = s.get("mean") or {}
        lines.append(
            f"| {s.get('method') or '-'} | {s.get('n_clips')} | {f(m.get('lsd'))} | "
            f"{f(m.get('mcd'))} | {f(m.get('speaker_cos'))} | {f(m.get('emotion_cos'))} | "
            f"{f(m.get('dnsmos_ovrl'))} | {f(m.get('dnsmos_p808'))} |"
        )
    return "\n".join(lines) + "\n"


def run_one_method(
    *,
    method: str,
    paired: list[tuple[str, str, str]],
    out_m: Path,
    args,
    codec_an: Path,
    dnsmos_dir: Path,
) -> dict:
    per_dir = out_m / "per_clip"
    logs_dir = out_m / "logs"
    per_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.summary_only):
        summary = aggregate(per_dir, method=method)
        dump_json(out_m / "summary.json", summary)
        (out_m / "FINDINGS.md").write_text(render_md([summary], title=f"{method} quality"), encoding="utf-8")
        return summary

    n_workers = max(1, int(args.num_workers))
    print(f"=== {method} n={len(paired)} workers={n_workers} -> {out_m} ===", flush=True)
    payloads = build_payloads(
        paired,
        num_workers=n_workers,
        npu_ids=parse_npu_ids(args.npu_ids),
        device_single=str(args.device),
        logs_dir=logs_dir,
        common={
            "method": method,
            "per_clip_dir": str(per_dir),
            "tmp_dir": str(out_m / "tmp"),
            "cpu_threads": int(args.cpu_threads),
            "resume": bool(args.resume),
            "seed": int(args.seed),
            "sr": int(args.sr),
            "codec_an": str(codec_an),
            "wavlm": args.wavlm,
            "emotion2vec": args.emotion2vec,
            "dnsmos_dir": str(dnsmos_dir),
        },
    )
    outs = run_shards(_worker, payloads, num_workers=n_workers)
    dump_json(out_m / "workers.json", outs)
    summary = aggregate(per_dir, method=method)
    dump_json(out_m / "summary.json", summary)
    (out_m / "FINDINGS.md").write_text(render_md([summary], title=f"{method} quality"), encoding="utf-8")
    print(
        f"done {method} n={summary['n_clips']} "
        f"err={sum(int(o.get('n_err') or 0) for o in outs)}",
        flush=True,
    )
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="wm_clean vs orig: LSD/MCD/spk/emo/DNSMOS")
    p.add_argument("--orig-dir", type=str, default="", help="原始语音目录")
    p.add_argument("--manifest", type=str, default="", help="优先于 --orig-dir")
    p.add_argument("--names", type=str, default="")
    p.add_argument(
        "--recon-root",
        type=str,
        required=True,
        help="如 .../fleurs/wm_n2000（下含 audioseal|band|wavmark/wav_wm_clean）",
    )
    p.add_argument("--wav-subdir", type=str, default=DEFAULT_WAV_SUBDIR)
    p.add_argument(
        "--methods",
        type=str,
        default="",
        help="逗号分隔；空则自动发现 audioseal/band/wavmark",
    )
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--sr", type=int, default=16000, help="统一重采样后再算指标")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--codec-an", type=str, default="")
    p.add_argument("--wavlm", type=str, default=first_existing(*_WAVLM))
    p.add_argument("--emotion2vec", type=str, default=first_existing(*_EMO))
    p.add_argument("--dnsmos-dir", type=str, default="")
    p.add_argument("--summary-only", action="store_true")
    add_npu_args(p, default_workers=8, default_device="npu:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_resume:
        args.resume = False

    recon_root = Path(args.recon_root)
    if not recon_root.is_dir():
        raise SystemExit(f"recon-root not found: {recon_root}")

    method_list = [x.strip() for x in str(args.methods).split(",") if x.strip()] or None
    methods = discover_methods(recon_root, str(args.wav_subdir), method_list)
    if not methods:
        raise SystemExit(f"no methods under {recon_root} with subdir {args.wav_subdir}")

    if str(args.manifest or "").strip():
        orig_clips = load_clips_from_manifest(Path(args.manifest), limit=int(args.limit))
        src_desc = f"manifest={args.manifest}"
    else:
        if not str(args.orig_dir or "").strip():
            raise SystemExit("need --orig-dir or --manifest")
        name_set = {x.strip() for x in str(args.names).replace(" ", ",").split(",") if x.strip()} or None
        orig_clips = list_clips(Path(args.orig_dir), limit=int(args.limit), names=name_set)
        src_desc = f"orig_dir={args.orig_dir}" + (f" names={len(name_set)}" if name_set else "")
    if not orig_clips:
        raise SystemExit(f"no original clips ({src_desc})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codec_an = find_eval_root(str(args.codec_an))
    dnsmos_dir = resolve_dnsmos_dir(str(args.dnsmos_dir))

    (out_dir / "RUN.md").write_text(
        "# wm_clean quality\n\n"
        f"- orig: `{src_desc}` n={len(orig_clips)}\n"
        f"- recon_root: `{recon_root}`\n"
        f"- wav_subdir: `{args.wav_subdir}`\n"
        f"- methods: {methods}\n"
        f"- sr: {args.sr}\n"
        f"- metrics: LSD, MCD, speaker_cos, emotion_cos, DNSMOS (vs original)\n",
        encoding="utf-8",
    )

    summaries: list[dict] = []
    for method in methods:
        wav_dir = recon_root / method / str(args.wav_subdir)
        if not wav_dir.is_dir():
            print(f"skip {method}: missing {wav_dir}", flush=True)
            continue
        paired = pair_clips(orig_clips, wav_dir)
        if not paired:
            print(f"skip {method}: 0 paired under {wav_dir}", flush=True)
            continue
        miss = len(orig_clips) - len(paired)
        if miss:
            print(f"{method}: paired={len(paired)} missing_recon={miss}", flush=True)
        summary = run_one_method(
            method=method,
            paired=paired,
            out_m=out_dir / method,
            args=args,
            codec_an=codec_an,
            dnsmos_dir=dnsmos_dir,
        )
        summaries.append(summary)

    dump_json(out_dir / "summary_all.json", {"methods": summaries, "metric_version": METRIC_VERSION})
    (out_dir / "FINDINGS.md").write_text(render_md(summaries), encoding="utf-8")
    print(f"all done methods={len(summaries)} -> {out_dir}", flush=True)


if __name__ == "__main__":
    ensure_spawn()
    main()
