"""宽松攻击（ATTACK_PROTOCOL §B）× {audioseal, wavmark, band_L9}。

**water_marks / P1 自包含**（不依赖 codec_an）。
攻击：``base.mild_attacks``（crop=0.7 / SNR=20 / speed=0.9 等）。
可 ``--wav-dir`` / ``--manifest`` / ``--names`` 指定语音，``--out-dir`` 指定输出。

例::

  python -m P1.experiments.run_mild_attack_three_n1500 \\
    --wav-dir /path/to/wavs --out-dir /path/to/out \\
    --methods audioseal,wavmark,band \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7

  python -m P1.experiments.run_mild_attack_three_n1500 \\
    --manifest /path/to/manifest.jsonl --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from base.channels import P1_BAND_L9, profile_score_cfg, score_wav_once
from base.config import EXP_ROOT, default_mix_1500, default_model_path, default_table_path, first_existing
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
from P1.experiments.run_anchor_attack_three_n1500 import (
    find_wm_wav,
    list_clips,
    load_clips_from_manifest,
    load_split,
)

OUT_STEM = "mild_attack_three_n1500"
_OUT_CANDS = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
    Path(r"E:\code\water_marks\result") / OUT_STEM,
    EXP_ROOT / OUT_STEM,
)
DEFAULT_OUT = next((p for p in _OUT_CANDS if p.parent.exists()), _OUT_CANDS[0])

# FAR01 冻结阈值（与 Anchor 三方法对照同口径）
T_WM = 1e-12
T_AS = 0.18587467074494227
T_ABA = 0.08451842905452735
T_SLIDE = 0.3238710454914386

SPLIT_CANDS = (
    Path(r"D:\data\codec_an\anas\band\attack_layer_gap_n1500_calib\random_500\split.json"),
    Path(r"/data1/home/ynian/codes/codec_an/anas/band/attack_layer_gap_n1500_calib/random_500/split.json"),
)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def process_band(model, sr, table, stem, wav_path, attacks, from_wm: Path | None) -> dict:
    wav = np.asarray(read_wav(str(wav_path), sr), dtype=np.float32).reshape(-1)
    if from_wm is not None:
        wm = np.asarray(read_wav(str(from_wm), sr), dtype=np.float32).reshape(-1)
        n = min(wav.shape[0], wm.shape[0])
        wav, wm = wav[:n], wm[:n]
    else:
        emb = embed_wav(model, wav, table, high_start=9, high_end=9, layers=[9], period=3)
        wm = np.asarray(emb["wav"], dtype=np.float32).reshape(-1)
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
    return {
        "name": stem,
        "method": "band",
        "clean_orig_aba": float(so["aba_embed"]),
        "clean_orig_slide": float(so["slide18"]),
        "clean_wm_aba": float(sw["aba_embed"]),
        "clean_wm_slide": float(sw["slide18"]),
        "attacks": atk,
        "sr": int(sr),
    }


def process_audioseal(gen, det, device, stem, wav_path, attacks, from_wm: Path | None) -> dict:
    wav16 = np.asarray(read_wav(str(wav_path), 16000), dtype=np.float32).reshape(-1)
    if from_wm is not None:
        wm16 = np.asarray(read_wav(str(from_wm), 16000), dtype=np.float32).reshape(-1)
    else:
        wm16, _ = audioseal_embed(gen, wav16, 16000, device)
    p_orig = audioseal_detect(det, wav16, 16000, device)
    p_wm = audioseal_detect(det, wm16, 16000, device)
    atk = {}
    for key, fn, meta in attacks:
        try:
            w2 = fn(wm16, 16000)
            atk[key] = {"wm_prob": audioseal_detect(det, w2, 16000, device), "meta": meta}
        except Exception as e:
            atk[key] = {"error": f"{type(e).__name__}: {e}", "meta": meta}
    return {
        "name": stem,
        "method": "audioseal",
        "clean_orig_prob": float(p_orig),
        "clean_wm_prob": float(p_wm),
        "attacks": atk,
        "sr": 16000,
    }


def process_wavmark(model, stem, wav_path, attacks, from_wm: Path | None) -> dict:
    import librosa

    wav_n, sr0 = librosa.load(str(wav_path), sr=None, mono=True)
    wav = np.asarray(wav_n, dtype=np.float32).reshape(-1)
    sr0 = int(sr0)
    payload = None
    if from_wm is not None:
        wm = np.asarray(librosa.load(str(from_wm), sr=sr0, mono=True)[0], dtype=np.float32).reshape(-1)
    else:
        wm, payload, _ = wavmark_embed(model, wav, sr0)
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
    return {
        "name": stem,
        "method": "wavmark",
        "clean_orig_prob": float(d_o["wm_prob"]),
        "clean_wm_prob": float(d_w["wm_prob"]),
        "attacks": atk,
        "sr": int(sr0),
    }


def _worker(payload: dict) -> dict:
    log_fp, log = open_worker_log(payload)
    wid = int(payload.get("worker_id", 0))
    method = str(payload["method"])
    device = str(payload["device"])
    out_pc = Path(payload["per_clip_dir"])
    resume = bool(payload.get("resume", True))
    n_ok = n_err = 0
    wm_dir = Path(payload["wm_dir"]) if payload.get("wm_dir") else None
    try:
        limit_cpu_threads(int(payload.get("cpu_threads", 1) or 1))
        pin_npu(device)
        set_seed(int(payload.get("seed", 0)) + wid)

        # reencode 需要 MOSS：band 本机已有；AS/WM 额外加载
        moss_for_atk = None
        band_model = table = sr = None
        as_gen = as_det = None
        wm_model = None
        if method == "band":
            band_model, sr = load_model(str(payload["moss_path"]), device)
            table = ensure_rare_table(str(payload["table"]))
            moss_for_atk = band_model
        elif method == "audioseal":
            disable_torch_compile_on_npu(device)
            as_gen, as_det = load_audioseal(
                str(payload["as_gen"]), str(payload["as_det"]), device, nbits=int(payload.get("nbits", 16))
            )
            if "reencode" in (payload.get("attack_keys") or DEFAULT_MILD_KEYS):
                moss_for_atk, _ = load_model(str(payload["moss_path"]), device)
        elif method == "wavmark":
            wm_model = load_wavmark(str(payload["wm_ckpt"]), device)
            if "reencode" in (payload.get("attack_keys") or DEFAULT_MILD_KEYS):
                moss_for_atk, _ = load_model(str(payload["moss_path"]), device)
        else:
            raise SystemExit(f"unknown method {method}")

        attacks = build_mild_attacks(
            payload.get("attack_keys"),
            moss_model=moss_for_atk,
            crop_keep=float(payload.get("crop_keep", 0.7)),
            snr_db=float(payload.get("snr_db", 20.0)),
            speed_factor=float(payload.get("speed_factor", 0.9)),
        )
        log(f"worker{wid} method={method} device={device} n={len(payload['items'])} attacks={len(attacks)}")

        for i, (stem, path) in enumerate(payload["items"]):
            dest = out_pc / f"{stem}.json"
            if resume and dest.is_file():
                n_ok += 1
                continue
            try:
                from_wm = find_wm_wav(stem, wm_dir)
                if method == "band":
                    rec = process_band(band_model, sr, table, stem, path, attacks, from_wm)
                elif method == "audioseal":
                    rec = process_audioseal(as_gen, as_det, device, stem, path, attacks, from_wm)
                else:
                    rec = process_wavmark(wm_model, stem, path, attacks, from_wm)
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


def merge_method(out_method: Path, method: str, calib: set[str], attack_keys: list[str]) -> dict:
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

    rows = []
    if method == "band":
        neg_a = [r["clean_orig_aba"] for r in hold]
        neg_s = [r["clean_orig_slide"] for r in hold]
        for key in attack_keys:
            pos_a, pos_s, pos_c = [], [], []
            for r in hold:
                blk = (r.get("attacks") or {}).get(key) or {}
                if "error" in blk:
                    continue
                pos_a.append(blk.get("aba_embed"))
                pos_s.append(blk.get("slide_v3"))
                pos_c.append(blk.get("channel_max"))
            # 级联：aba≥T_aba ∨ slide≥T_slide
            tpr_hits = far_hits = 0
            n_pos = n_neg = 0
            for r in hold:
                blk = (r.get("attacks") or {}).get(key) or {}
                if "error" in blk or blk.get("aba_embed") is None:
                    continue
                n_pos += 1
                if float(blk["aba_embed"]) >= T_ABA or float(blk["slide_v3"]) >= T_SLIDE:
                    tpr_hits += 1
            for a, s in zip(neg_a, neg_s):
                n_neg += 1
                if float(a) >= T_ABA or float(s) >= T_SLIDE:
                    far_hits += 1
            rows.append(
                {
                    "attack": key,
                    "tpr": tpr_hits / n_pos if n_pos else None,
                    "far": far_hits / n_neg if n_neg else None,
                    "cascade_tpr": tpr_hits / n_pos if n_pos else None,
                    "cascade_far": far_hits / n_neg if n_neg else None,
                    "mean_aba": float(np.nanmean([x for x in pos_a if x is not None])) if pos_a else None,
                    "mean_slide": float(np.nanmean([x for x in pos_s if x is not None])) if pos_s else None,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                }
            )
    else:
        T = T_AS if method == "audioseal" else T_WM
        neg = [r["clean_orig_prob"] for r in hold]
        for key in attack_keys:
            pos = []
            for r in hold:
                blk = (r.get("attacks") or {}).get(key) or {}
                if "error" in blk:
                    continue
                pos.append(blk.get("wm_prob"))
            rows.append(
                {
                    "attack": key,
                    "tpr": rate(pos, T),
                    "far": rate(neg, T),
                    "mean_wm_prob": float(np.nanmean([x for x in pos if x is not None])) if pos else None,
                    "n_pos": len([x for x in pos if x is not None]),
                    "n_neg": len(neg),
                    "T": T,
                }
            )
    return {
        "method": method,
        "n_holdout": len(hold),
        "n_files": len(files),
        "rows": rows,
    }


def write_compare_md(out_root: Path, by_method: dict, attack_keys: list[str]) -> None:
    lines = [
        "# Mild attack three methods (宽松)",
        "",
        "| attack | WavMark TPR/FAR | AudioSeal TPR/FAR | band 级联 TPR/FAR |",
        "|---|---|---|---|",
    ]

    def pct(x):
        return "—" if x is None else f"{100 * float(x):.2f}%"

    wm = {r["attack"]: r for r in (by_method.get("wavmark") or {}).get("rows") or []}
    as_ = {r["attack"]: r for r in (by_method.get("audioseal") or {}).get("rows") or []}
    bd = {r["attack"]: r for r in (by_method.get("band") or {}).get("rows") or []}
    for key in attack_keys:
        w, a, b = wm.get(key, {}), as_.get(key, {}), bd.get(key, {})
        lines.append(
            f"| {key} | {pct(w.get('tpr'))}/{pct(w.get('far'))} | "
            f"{pct(a.get('tpr'))}/{pct(a.get('far'))} | "
            f"{pct(b.get('cascade_tpr'))}/{pct(b.get('cascade_far'))} |"
        )
    (out_root / "COMPARE_MILD_FAR01.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_method(args, method: str, clips: list[tuple[str, str]], out_root: Path, attack_keys: list[str]) -> dict:
    out_m = out_root / method
    for sub in ("per_clip", "errors", "logs"):
        (out_m / sub).mkdir(parents=True, exist_ok=True)
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
        "attack_keys": list(attack_keys),
        "wm_dir": str(wm_map.get(method) or "") or None,
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
    if not common["wm_dir"]:
        common["wm_dir"] = None
    n_workers = max(1, int(args.num_workers))
    print(f"=== {method} workers={n_workers} clips={len(clips)} mild_attacks={len(attack_keys)} ===", flush=True)
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
    return merge_method(out_m, method, calib, attack_keys)


def main() -> None:
    ensure_spawn()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav-dir", type=str, default=default_mix_1500(), help="语音目录（递归）")
    p.add_argument("--manifest", type=str, default="", help="manifest.json/.jsonl 或路径列表（优先）")
    p.add_argument("--names", type=str, default="", help="stem 白名单，逗号分隔")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--methods", type=str, default="audioseal,wavmark,band")
    p.add_argument("--attacks", type=str, default=",".join(DEFAULT_MILD_KEYS), help="宽松攻击 key 列表")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--split-json", type=str, default=str(first_existing(*SPLIT_CANDS, "") or ""))
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--table", type=str, default=default_table_path())
    p.add_argument("--audioseal-generator", type=str, default=first_file(*AS_GEN_CANDIDATES))
    p.add_argument("--audioseal-detector", type=str, default=first_file(*AS_DET_CANDIDATES))
    p.add_argument("--wavmark-checkpoint", type=str, default=first_file(*WM_CKPT_CANDIDATES))
    p.add_argument("--audioseal-wm-dir", type=str, default="")
    p.add_argument("--wavmark-wm-dir", type=str, default="")
    p.add_argument("--band-wm-dir", type=str, default="")
    p.add_argument("--crop-keep", type=float, default=0.7)
    p.add_argument("--snr-db", type=float, default=20.0)
    p.add_argument("--speed-factor", type=float, default=0.9)
    add_npu_args(p, default_workers=8, default_device="npu:0")
    args = p.parse_args()
    if args.no_resume:
        args.resume = False

    attack_keys = [x.strip() for x in str(args.attacks).split(",") if x.strip()]
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
        "# Mild attack three methods (宽松)\n\n"
        f"- source: `{src_desc}`\n"
        f"- n_clips: {len(clips)}\n"
        f"- methods: {methods}\n"
        f"- attacks: {attack_keys}\n"
        f"- crop_keep={args.crop_keep} snr_db={args.snr_db} speed={args.speed_factor}\n"
        f"- workers: {args.num_workers}  npu_ids: {args.npu_ids}\n"
        f"- out: `{out_root}`\n",
        encoding="utf-8",
    )

    summaries = {}
    t0 = time.time()
    for m in methods:
        summaries[m] = run_one_method(args, m, clips, out_root, attack_keys)
    write_compare_md(out_root, summaries, attack_keys)
    dump_json(out_root / "all_summaries.json", summaries)
    print(f"DONE {out_root} elapsed={((time.time() - t0) / 60):.1f} min", flush=True)
    print("compare:", out_root / "COMPARE_MILD_FAR01.md", flush=True)


if __name__ == "__main__":
    main()
