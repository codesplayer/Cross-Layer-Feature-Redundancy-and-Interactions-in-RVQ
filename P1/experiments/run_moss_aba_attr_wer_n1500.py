"""对 ABA 实验保存的语音计算说话人余弦和情感余弦。

参照是 mix_1500 里同一句的原始语音，不是干净重构。
说话人：WavLM-SV。情感：emotion2vec。本轮不算 WER。
``clean`` 也相对原始语音计算，用来看编解码本身的变化。

8 进程 × 8 NPU。每条条件一个 JSON：``{out}/{clean|single|pair|band_L04_L14|skip}/.../{stem}.json``。

例::

  python -m P1.experiments.run_moss_aba_attr_wer_n1500 \\
    --wav-dir /data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500 \\
    --wav-root /data1/home/ynian/codes/eval/add_exp/result/moss_aba_emotion_wav_n1500 \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_EVAL_REL = Path("eval") / "experiments" / "calcu_emotion_sim.py"


def find_eval_root(explicit: str = "") -> Path:
    """返回包含 ``eval/experiments/calcu_emotion_sim.py`` 的目录。

    服务器上 ``/data1/home/ynian/codes/codec_an`` 只是数据根，包在
    ``exp_watermark/codec_an`` 下面。只判断目录存在会把错误的根加进 sys.path。
    """
    here = Path(__file__).resolve()
    candidates: list[str] = [
        explicit,
        os.environ.get("CODEC_AN", ""),
        r"E:\code\codec_an",
        r"/data1/home/ynian/codes/codec_an/exp_watermark/codec_an",
        r"/data1/home/ynian/codes/codec_an",
        r"/data1/home/ynian/codes/eval/add_exp/codec_an",
        r"/data1/home/ynian/codes/eval/codec_an",
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
        if (Path(root) / _EVAL_REL).is_file():
            return Path(root)
    raise SystemExit(
        "找不到 eval.experiments.calcu_emotion_sim。"
        "请用 --codec-an 或环境变量 CODEC_AN 指向含 eval/ 的目录。"
    )


def use_eval_root(root: Path) -> None:
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)

from base.config import WAVLM_SV_PATH, default_mix_1500, first_existing
from base.util import AUDIO_EXTS
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

GROUPS = ("clean", "single", "pair", "band_L04_L14", "skip")
_WAVLM = (
    Path(WAVLM_SV_PATH),
    Path(r"/data1/home/ynian/pred_models/wavlm-base-plus-sv"),
    Path(r"E:\code\codec_an\model\wavlm-base-plus-sv"),
)
_EMO = (
    Path(r"/data1/home/ynian/pred_models/emotion2vec_plus_large"),
    Path(r"E:\code\codec_an\model\emotion2vec_plus_large"),
)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


_EXT_RANK = {".wav": 0, ".flac": 1, ".mp3": 2, ".ogg": 3, ".m4a": 4, ".mp4": 5}


def iter_audio(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def index_originals(wav_dir: Path) -> dict[str, Path]:
    """按文件名对齐。原始语音可以是 wav 或 mp3。同一句两种都有时用 wav。"""
    found: dict[str, Path] = {}
    rank: dict[str, int] = {}
    for path in iter_audio(wav_dir):
        ext = path.suffix.lower()
        score = _EXT_RANK.get(ext, 9)
        if path.stem not in found or score < rank[path.stem]:
            found[path.stem] = path
            rank[path.stem] = score
    return found


def list_jobs(wav_root: Path, originals: dict[str, Path]) -> list[dict]:
    jobs = []
    missing = 0
    for group in GROUPS:
        root = wav_root / group
        if not root.exists():
            continue
        if group in ("clean", "band_L04_L14"):
            cond_wavs = [(group, wav) for wav in iter_audio(root)]
        else:
            cond_wavs = [
                (cond_dir.name, wav)
                for cond_dir in sorted(p for p in root.iterdir() if p.is_dir())
                for wav in iter_audio(cond_dir)
            ]
        for cond, wav in cond_wavs:
            ref = originals.get(wav.stem)
            if ref is None:
                missing += 1
                continue
            jobs.append(
                {
                    "stem": wav.stem,
                    "group": group,
                    "cond": cond,
                    "wav": str(wav),
                    "ref": str(ref),
                }
            )
    if missing:
        print(f"skip {missing} wavs with no original", flush=True)
    return jobs


def worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads", 2)))
    device = str(payload["device"])
    pin_npu(device)
    log_fp, log = open_worker_log(payload)
    n_ok = n_err = 0
    try:
        use_eval_root(Path(payload["codec_an"]))
        from eval.experiments.calcu_emotion_sim import CalcuEmotionSim
        from eval.experiments.calcu_speaker_sim import CalcuSpeakerSim

        spk = CalcuSpeakerSim(model_source=payload["wavlm"], device=device)
        emo = CalcuEmotionSim(model_id=payload["emotion2vec"], device=device)
        out_dir = Path(payload["out_dir"])
        for job in payload["items"]:
            if job["group"] in ("clean", "band_L04_L14"):
                dest = out_dir / job["group"] / f"{job['stem']}.json"
            else:
                dest = out_dir / job["group"] / job["cond"] / f"{job['stem']}.json"
            if dest.exists():
                n_ok += 1
                continue
            try:
                rec = {
                    "stem": job["stem"],
                    "group": job["group"],
                    "cond": job["cond"],
                    "ref": job["ref"],
                    "speaker_cos": float(spk.sim_file(job["ref"], job["wav"])),
                    "emotion_cos": float(emo.sim_file(job["ref"], job["wav"])),
                }
                write_json(dest, rec)
                n_ok += 1
                if n_ok % 20 == 0:
                    log(f"ok={n_ok} {job['cond']} {job['stem']}")
            except Exception as exc:
                n_err += 1
                log(f"ERR {job['cond']} {job['stem']}: {exc}")
        free_device()
    finally:
        if log_fp is not None:
            log_fp.close()
    return {"worker_id": payload.get("worker_id"), "n_ok": n_ok, "n_error": n_err}


def main() -> None:
    ensure_spawn()
    p = argparse.ArgumentParser(description=__doc__)
    add_npu_args(p)
    p.add_argument("--wav-dir", type=str, default=default_mix_1500(), help="原始语音目录，按文件名与生成 wav 对齐")
    p.add_argument("--codec-an", type=str, default="", help="含 eval/ 的 codec_an 根目录")
    p.add_argument(
        "--wav-root",
        type=str,
        default=r"/data1/home/ynian/codes/eval/add_exp/result/moss_aba_emotion_wav_n1500",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=r"/data1/home/ynian/codes/eval/add_exp/result/moss_aba_attr_n1500",
    )
    p.add_argument("--wavlm", type=str, default=first_existing(*_WAVLM))
    p.add_argument("--emotion2vec", type=str, default=first_existing(*_EMO))
    args = p.parse_args()
    codec_an = find_eval_root(str(args.codec_an))
    use_eval_root(codec_an)
    print(f"codec_an={codec_an}", flush=True)
    originals = index_originals(Path(args.wav_dir))
    if not originals:
        raise SystemExit(f"no original wav in {args.wav_dir}")
    jobs = list_jobs(Path(args.wav_root), originals)
    if not jobs:
        raise SystemExit(f"no wav pairs under {args.wav_root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"jobs={len(jobs)} -> {out_dir}", flush=True)
    payloads = build_payloads(
        jobs,
        num_workers=int(args.num_workers),
        npu_ids=parse_npu_ids(args.npu_ids),
        device_single=str(args.device),
        logs_dir=out_dir / "logs",
        common={
            "out_dir": str(out_dir),
            "cpu_threads": int(args.cpu_threads),
            "codec_an": str(codec_an),
            "wavlm": args.wavlm,
            "emotion2vec": args.emotion2vec,
        },
    )
    outs = run_shards(worker, payloads, num_workers=int(args.num_workers))
    dump_json(out_dir / "launch_summary.json", {"n_jobs": len(jobs), "workers": outs})
    print("done", outs, flush=True)


if __name__ == "__main__":
    main()
