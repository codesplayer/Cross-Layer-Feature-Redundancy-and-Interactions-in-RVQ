"""对 audioseal / wavmark / L9_clean / codec_recon 四套语音算 WER、CER。

只跑这四个目录里的目标 wav（不做攻击子集）：
- audioseal/wav_wm
- wavmark/wav_wm
- L9_clean/wav_wm
- codec_recon_n1500/wav_moss

语种与模型同 ABA 脚本：aishell / sub_ZH → Paraformer；sub_EN → Whisper。
参照文本：mix_1500_audio_text.json（按文件名对齐）。

8 进程 × 8 NPU。输出 ``{out}/{set_name}/{stem}.json``。

例::

  python -m P1.experiments.run_baselines_asr_wer_n1500 \\
    --audio-root D:/data/codec_an/eval/eval/audios \\
    --text-json D:/data/codec_an/data/mix_1500_audio_text.json \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
from pathlib import Path

from base.config import first_existing
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
from base.util import AUDIO_EXTS, dump_json
from P1.experiments.run_moss_aba_asr_wer_n1500 import (
    _ASR_PY,
    _PARAFORMER,
    _TEXT_JSON,
    _WER_PY,
    _WHISPER,
    build_asr,
    lang_from_stem,
    load_module,
    load_ref_text,
    process_for_cer,
    process_for_wer,
    write_json,
)

# set_name -> 相对 audio-root 的语音目录
SETS = (
    ("audioseal", "audioseal/wav_wm"),
    ("wavmark", "wavmark/wav_wm"),
    ("L9_clean", "L9_clean/wav_wm"),
    ("codec_recon_n1500", "codec_recon_n1500/wav_moss"),
)


def iter_audio(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def list_jobs(audio_root: Path, refs: dict[str, dict]) -> list[dict]:
    jobs = []
    missing = 0
    for set_name, rel in SETS:
        root = audio_root / rel
        if not root.is_dir():
            print(f"missing set dir: {root}", flush=True)
            continue
        for wav in iter_audio(root):
            ref = refs.get(wav.stem)
            if ref is None or not str(ref.get("text") or "").strip():
                missing += 1
                continue
            lang = str(ref.get("language") or lang_from_stem(wav.stem))
            jobs.append(
                {
                    "stem": wav.stem,
                    "group": set_name,
                    "cond": set_name,
                    "wav": str(wav),
                    "text": ref["text"],
                    "language": lang,
                    "text_id": ref.get("text_id"),
                }
            )
    if missing:
        print(f"skip {missing} wavs with no ref text", flush=True)
    return jobs


def dest_path(out_dir: Path, job: dict) -> Path:
    return out_dir / job["group"] / f"{job['stem']}.json"


def worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads", 2)))
    device = str(payload["device"])
    pin_npu(device)
    log_fp, log = open_worker_log(payload)
    n_ok = n_err = 0
    try:
        wer_mod = load_module(payload["wer_py"], f"baseline_wer_{payload.get('worker_id', 0)}")
        from jiwer import process_characters, process_words

        tn = wer_mod.TextNormalizer()
        asr = build_asr(device, payload["whisper"], payload["paraformer"], payload["asr_py"])
        out_dir = Path(payload["out_dir"])
        for job in payload["items"]:
            dest = dest_path(out_dir, job)
            if dest.exists():
                n_ok += 1
                continue
            try:
                lang = job["language"]
                model_name = "whisper" if lang == "en" else "paraformer"
                hyp = asr.asr(job["wav"], lang=lang)
                ref = job["text"]
                ref_w = process_for_wer(tn, ref, lang)
                hyp_w = process_for_wer(tn, hyp, lang)
                ref_c = process_for_cer(tn, ref, lang)
                hyp_c = process_for_cer(tn, hyp, lang)
                wer = float(process_words(ref_w, hyp_w).wer) if ref_w.strip() else None
                cer = float(process_characters(ref_c, hyp_c).cer) if ref_c.strip() else None
                write_json(
                    dest,
                    {
                        "stem": job["stem"],
                        "set": job["group"],
                        "language": lang,
                        "asr_model": model_name,
                        "text_id": job.get("text_id"),
                        "ref": ref,
                        "hyp": hyp,
                        "wer": wer,
                        "cer": cer,
                        "wav": job["wav"],
                    },
                )
                n_ok += 1
                if n_ok % 20 == 0:
                    log(f"ok={n_ok} {job['group']} {job['stem']} {model_name} wer={wer} cer={cer}")
            except Exception as exc:
                n_err += 1
                log(f"ERR {job['group']} {job['stem']}: {exc}")
        free_device()
    finally:
        if log_fp is not None:
            log_fp.close()
    return {"worker_id": payload.get("worker_id"), "n_ok": n_ok, "n_error": n_err}


def main() -> None:
    ensure_spawn()
    p = argparse.ArgumentParser(description=__doc__)
    add_npu_args(p)
    p.add_argument(
        "--audio-root",
        type=str,
        default=first_existing(
            Path(r"D:\data\codec_an\eval\eval\audios"),
            Path(r"/data1/home/ynian/codes/eval/audios"),
            Path(r"/data1/home/ynian/codes/codec_an/eval/eval/audios"),
        ),
    )
    p.add_argument("--text-json", type=str, default=first_existing(*_TEXT_JSON))
    p.add_argument(
        "--out_dir",
        type=str,
        default=r"D:\data\codec_an\eval\eval\audios\result\baselines_asr_wer_n1500",
    )
    p.add_argument("--whisper", type=str, default=first_existing(*_WHISPER))
    p.add_argument("--paraformer", type=str, default=first_existing(*_PARAFORMER))
    p.add_argument("--asr-py", type=str, default=first_existing(*_ASR_PY))
    p.add_argument("--wer-py", type=str, default=first_existing(*_WER_PY))
    args = p.parse_args()

    for label, path in (("text-json", args.text_json), ("asr-py", args.asr_py), ("wer-py", args.wer_py)):
        if not Path(path).exists():
            raise SystemExit(f"missing {label}: {path}")

    refs = load_ref_text(Path(args.text_json))
    jobs = list_jobs(Path(args.audio_root), refs)
    if not jobs:
        raise SystemExit(f"no jobs under {args.audio_root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_set: dict[str, int] = {}
    for j in jobs:
        by_set[j["group"]] = by_set.get(j["group"], 0) + 1
    n_zh = sum(1 for j in jobs if j["language"] == "zh")
    n_en = sum(1 for j in jobs if j["language"] == "en")
    print(
        f"jobs={len(jobs)} zh={n_zh} en={n_en} by_set={by_set}\n"
        f"  audio-root={args.audio_root}\n  text={args.text_json}\n"
        f"  whisper={args.whisper}\n  paraformer={args.paraformer}\n  -> {out_dir}",
        flush=True,
    )
    payloads = build_payloads(
        jobs,
        num_workers=int(args.num_workers),
        npu_ids=parse_npu_ids(args.npu_ids),
        device_single=str(args.device),
        logs_dir=out_dir / "logs",
        common={
            "out_dir": str(out_dir),
            "cpu_threads": int(args.cpu_threads),
            "whisper": args.whisper,
            "paraformer": args.paraformer,
            "asr_py": args.asr_py,
            "wer_py": args.wer_py,
        },
    )
    outs = run_shards(worker, payloads, num_workers=int(args.num_workers))
    dump_json(
        out_dir / "launch_summary.json",
        {"n_jobs": len(jobs), "n_zh": n_zh, "n_en": n_en, "by_set": by_set, "workers": outs},
    )
    print("done", outs, flush=True)


if __name__ == "__main__":
    main()
