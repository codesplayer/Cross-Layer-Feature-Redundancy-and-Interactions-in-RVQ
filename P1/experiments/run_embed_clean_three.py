"""三方法干净嵌印：只加水印、保存语音（不做攻击/检测汇总）。

输出约定::

  {out}/{method}/wav_wm_clean/{stem}_wm.wav
  method ∈ {audioseal, band, wavmark}

例::

  python -m P1.experiments.run_embed_clean_three \\
    --wav-dir /data1/.../fleurs/select_n2000 \\
    --out-dir /data1/.../fleurs/wm_n2000 \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from base.config import (
    EXP_ROOT,
    default_mix_1500,
    default_model_path,
    default_table_path,
)
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
from base.util import dump_json, embed_wav, load_model
from P1.baselines import (
    AS_DET_CANDIDATES,
    AS_GEN_CANDIDATES,
    WM_CKPT_CANDIDATES,
    audioseal_embed,
    disable_torch_compile_on_npu,
    first_file,
    load_audioseal,
    load_wavmark,
    wavmark_embed,
)
from P1.experiments.run_anchor_attack_three_n1500 import (
    list_clips,
    load_clips_from_manifest,
    save_clean_wm_wav,
)

OUT_STEM = "wm_clean_embed"
_OUT_CANDS = (
    Path(r"/data1/home/ynian/codes/data/fleurs") / "wm_n2000",
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
    Path(r"E:\code\water_marks\result") / OUT_STEM,
    EXP_ROOT / OUT_STEM,
)
DEFAULT_OUT = next((p for p in _OUT_CANDS if p.parent.exists()), _OUT_CANDS[0])
DEFAULT_METHODS = ("audioseal", "band", "wavmark")
WAV_SUBDIR = "wav_wm_clean"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def embed_band(model, sr, table, stem: str, wav_path: str, clean_wm_dir: Path) -> dict:
    wav = np.asarray(read_wav(str(wav_path), sr), dtype=np.float32).reshape(-1)
    emb = embed_wav(model, wav, table, high_start=9, high_end=9, layers=[9], period=3)
    wm = np.asarray(emb["wav"], dtype=np.float32).reshape(-1)
    path = save_clean_wm_wav(clean_wm_dir, stem, wm, int(sr))
    return {"name": stem, "method": "band", "sr": int(sr), "clean_wm_wav": path, "n_samples": int(wm.shape[0])}


def embed_audioseal(gen, device, stem: str, wav_path: str, clean_wm_dir: Path) -> dict:
    wav16 = np.asarray(read_wav(str(wav_path), 16000), dtype=np.float32).reshape(-1)
    wm16, _ = audioseal_embed(gen, wav16, 16000, device)
    path = save_clean_wm_wav(clean_wm_dir, stem, wm16, 16000)
    return {
        "name": stem,
        "method": "audioseal",
        "sr": 16000,
        "clean_wm_wav": path,
        "n_samples": int(wm16.shape[0]),
    }


def embed_wavmark_one(model, stem: str, wav_path: str, clean_wm_dir: Path) -> dict:
    import librosa

    wav_n, sr0 = librosa.load(str(wav_path), sr=None, mono=True)
    wav = np.asarray(wav_n, dtype=np.float32).reshape(-1)
    sr0 = int(sr0)
    wm, _payload, _ = wavmark_embed(model, wav, sr0)
    path = save_clean_wm_wav(clean_wm_dir, stem, wm, sr0)
    return {
        "name": stem,
        "method": "wavmark",
        "sr": sr0,
        "clean_wm_wav": path,
        "n_samples": int(np.asarray(wm).reshape(-1).shape[0]),
    }


def _worker(payload: dict) -> dict:
    log_fp, log = open_worker_log(payload)
    wid = int(payload.get("worker_id", 0))
    method = str(payload["method"])
    device = str(payload["device"])
    out_pc = Path(payload["per_clip_dir"])
    clean_wm_dir = Path(payload["clean_wm_dir"])
    resume = bool(payload.get("resume", True))
    n_ok = n_skip = n_err = 0
    try:
        limit_cpu_threads(int(payload.get("cpu_threads", 1) or 1))
        pin_npu(device)
        set_seed(int(payload.get("seed", 0)) + wid)
        clean_wm_dir.mkdir(parents=True, exist_ok=True)
        out_pc.mkdir(parents=True, exist_ok=True)
        log(f"worker{wid} method={method} device={device} n={len(payload['items'])}")

        band_model = table = sr = None
        as_gen = None
        wm_model = None
        if method == "band":
            band_model, sr = load_model(str(payload["moss_path"]), device)
            table = ensure_rare_table(str(payload["table"]))
        elif method == "audioseal":
            disable_torch_compile_on_npu(device)
            as_gen, _ = load_audioseal(
                str(payload["as_gen"]),
                str(payload["as_det"]),
                device,
                nbits=int(payload.get("nbits", 16)),
            )
        elif method == "wavmark":
            wm_model = load_wavmark(str(payload["wm_ckpt"]), device)
        else:
            raise SystemExit(f"unknown method {method}")

        for i, (stem, path) in enumerate(payload["items"]):
            dest_json = out_pc / f"{stem}.json"
            dest_wav = clean_wm_dir / f"{stem}_wm.wav"
            if resume and dest_wav.is_file() and dest_json.is_file():
                n_skip += 1
                n_ok += 1
                continue
            try:
                if method == "band":
                    rec = embed_band(band_model, sr, table, stem, path, clean_wm_dir)
                elif method == "audioseal":
                    rec = embed_audioseal(as_gen, device, stem, path, clean_wm_dir)
                else:
                    rec = embed_wavmark_one(wm_model, stem, path, clean_wm_dir)
                _write_json(dest_json, rec)
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
            "method": method,
            "n_ok": n_ok,
            "n_skip": n_skip,
            "n_error": n_err,
        }
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        return {
            "worker_id": wid,
            "method": method,
            "n_ok": n_ok,
            "n_skip": n_skip,
            "n_error": n_err + 1,
            "error": str(e),
        }
    finally:
        free_device()
        if log_fp:
            log_fp.close()


def run_one_method(args, method: str, clips: list[tuple[str, str]], out_root: Path) -> dict:
    out_m = out_root / method
    clean_wm_dir = out_m / WAV_SUBDIR
    for sub in ("per_clip", "errors", "logs"):
        (out_m / sub).mkdir(parents=True, exist_ok=True)
    clean_wm_dir.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, int(args.num_workers))
    print(f"=== {method} embed workers={n_workers} clips={len(clips)} -> {clean_wm_dir} ===", flush=True)
    payloads = build_payloads(
        clips,
        num_workers=n_workers,
        npu_ids=parse_npu_ids(args.npu_ids),
        device_single=str(args.device),
        logs_dir=out_m / "logs",
        common={
            "method": method,
            "per_clip_dir": str(out_m / "per_clip"),
            "clean_wm_dir": str(clean_wm_dir),
            "resume": bool(args.resume),
            "seed": int(args.seed),
            "cpu_threads": int(args.cpu_threads),
            "moss_path": str(args.moss_path),
            "table": str(args.table),
            "as_gen": str(args.audioseal_generator),
            "as_det": str(args.audioseal_detector),
            "wm_ckpt": str(args.wavmark_checkpoint),
            "nbits": 16,
        },
    )
    outs = run_shards(_worker, payloads, num_workers=n_workers)
    dump_json(out_m / "workers.json", outs)
    n_wav = len(list(clean_wm_dir.glob("*_wm.wav")))
    summary = {
        "method": method,
        "n_clips": len(clips),
        "n_wm_wav": n_wav,
        "wav_wm_clean": str(clean_wm_dir.resolve()),
        "workers": outs,
    }
    dump_json(out_m / "summary.json", summary)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="三方法干净嵌印并保存 wav")
    p.add_argument("--wav-dir", type=str, default="", help="输入语音目录（递归）")
    p.add_argument("--manifest", type=str, default="", help="优先于 --wav-dir")
    p.add_argument("--names", type=str, default="")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT),
        help="输出根目录 → {out}/{method}/wav_wm_clean/",
    )
    p.add_argument("--methods", type=str, default="audioseal,band,wavmark")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--table", type=str, default=default_table_path())
    p.add_argument("--audioseal-generator", type=str, default=first_file(*AS_GEN_CANDIDATES))
    p.add_argument("--audioseal-detector", type=str, default=first_file(*AS_DET_CANDIDATES))
    p.add_argument("--wavmark-checkpoint", type=str, default=first_file(*WM_CKPT_CANDIDATES))
    add_npu_args(p, default_workers=8, default_device="npu:0")
    return p.parse_args()


def main() -> None:
    ensure_spawn()
    args = parse_args()
    if args.no_resume:
        args.resume = False

    if str(args.manifest or "").strip():
        clips = load_clips_from_manifest(Path(args.manifest), limit=int(args.limit))
        src_desc = f"manifest={args.manifest}"
    else:
        wav_dir = str(args.wav_dir or "").strip() or default_mix_1500()
        name_set = {x.strip() for x in str(args.names).replace(" ", ",").split(",") if x.strip()} or None
        clips = list_clips(Path(wav_dir), limit=int(args.limit), names=name_set)
        src_desc = f"wav_dir={wav_dir}" + (f" names={len(name_set)}" if name_set else "")
    if not clips:
        raise SystemExit(f"no clips ({src_desc})")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in DEFAULT_METHODS:
            raise SystemExit(f"unknown method {m}; choose from {DEFAULT_METHODS}")

    (out_root / "RUN.md").write_text(
        "# clean embed three methods\n\n"
        f"- input: `{src_desc}`\n"
        f"- n_clips: {len(clips)}\n"
        f"- methods: {methods}\n"
        f"- output: `{{out}}/{{method}}/{WAV_SUBDIR}/{{stem}}_wm.wav`\n"
        f"- workers: {args.num_workers}  npu_ids: {args.npu_ids}\n",
        encoding="utf-8",
    )

    summaries = {}
    t0 = time.time()
    for m in methods:
        summaries[m] = run_one_method(args, m, clips, out_root)
    dump_json(out_root / "summary_all.json", summaries)
    print(f"DONE {out_root} elapsed={((time.time() - t0) / 60):.1f} min", flush=True)
    for m, s in summaries.items():
        print(f"  {m}: n_wm={s.get('n_wm_wav')} -> {s.get('wav_wm_clean')}", flush=True)


if __name__ == "__main__":
    main()
