"""对 ABA 75 组语音做 ASR，相对 mix_1500_audio_text 计算 WER / CER。

语种与模型（沿用 ``codec_an/tools/experiments/ASR.py``）
------------------------------------------------
- aishell_*、sub_ZH_* → 中文 → Paraformer
- sub_EN_* → 英文 → Whisper-large-v3

文本规范化与 WER 分词沿用 ``codec_an/eval/experiments/calcu_wer.py`` 的
``TextNormalizer``：中文按字、英文按词。CER 在同一套规范化后按字符计算。

8 进程 × 8 NPU。每条条件一个 JSON：
``{out}/{group}/{cond}/{stem}.json``（clean / band 无 cond 子目录）。

例::

  python -m P1.experiments.run_moss_aba_asr_wer_n1500 \\
    --wav-root /data1/home/ynian/codes/eval/add_exp/result/moss_aba_emotion_wav_n1500 \\
    --text-json /data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500_audio_text.json \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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

GROUPS = ("clean", "single", "pair", "band_L04_L14", "skip")

EXP_ROOT = Path(r"/data1/home/ynian/codes/eval/add_exp/result")

_WHISPER = (
    Path(r"/data1/home/ynian/pred_models/whisper-large-v3"),
    Path(r"/home/ma-user/work/ynian/pred_model/whisper-large-v3"),
    Path(r"E:\pred_model\whisper-large-v3"),
)
_PARAFORMER = (
    Path(r"/data1/home/ynian/pred_models/paraformer_zh"),
    Path(r"/home/ma-user/work/ynian/pred_model/paraformer_zh"),
    Path(r"E:\pred_model\paraformer_zh"),
)
_TEXT_JSON = (
    Path(r"/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500_audio_text.json"),
    Path(r"/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500_audio_text.jsonl"),
    Path(r"D:\data\codec_an\data\mix_1500_audio_text.json"),
    Path(r"E:\code\codec_an\exp_watermark\data\mix_1500_audio_text.json"),
)
_ASR_PY = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/water_marks/eval/experiments/ASR.py"),
    Path(r"/data1/home/ynian/codes/codec_an/tools/experiments/ASR.py"),
    Path(r"E:\code\codec_an\tools\experiments\ASR.py"),
)
_WER_PY = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/water_marks/eval/experiments/calcu_wer.py"),
    Path(r"/data1/home/ynian/codes/codec_an/eval/experiments/calcu_wer.py"),
    Path(r"E:\code\codec_an\eval\experiments\calcu_wer.py"),
)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ref_text(path: Path) -> dict[str, dict]:
    """stem -> {text, language, text_id}。支持 json 数组或 jsonl。"""
    text = path.read_text(encoding="utf-8").strip()
    rows: list[dict]
    if path.suffix.lower() == ".jsonl" or (text and text[0] != "["):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    out = {}
    for r in rows:
        stem = Path(str(r.get("id") or r.get("audio") or "")).stem
        if not stem:
            continue
        lang = str(r.get("language") or "").lower()
        if not lang:
            lang = lang_from_stem(stem)
        out[stem] = {
            "text": str(r.get("text") or ""),
            "language": lang,
            "text_id": r.get("text_id"),
        }
    return out


def lang_from_stem(stem: str) -> str:
    if stem.startswith("sub_EN_") or "_EN_" in stem:
        return "en"
    return "zh"


def iter_audio(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def list_jobs(wav_root: Path, refs: dict[str, dict]) -> list[dict]:
    jobs = []
    missing = 0
    for group in GROUPS:
        root = wav_root / group
        if not root.exists():
            continue
        if group in ("clean", "band_L04_L14"):
            items = [(group, wav) for wav in iter_audio(root)]
        else:
            items = [
                (cond.name, wav)
                for cond in sorted(p for p in root.iterdir() if p.is_dir())
                for wav in iter_audio(cond)
            ]
        for cond, wav in items:
            ref = refs.get(wav.stem)
            if ref is None or not ref["text"].strip():
                missing += 1
                continue
            jobs.append(
                {
                    "stem": wav.stem,
                    "group": group,
                    "cond": cond,
                    "wav": str(wav),
                    "text": ref["text"],
                    "language": ref["language"],
                    "text_id": ref.get("text_id"),
                }
            )
    if missing:
        print(f"skip {missing} wavs with no ref text", flush=True)
    return jobs


def dest_path(out_dir: Path, job: dict) -> Path:
    if job["group"] in ("clean", "band_L04_L14"):
        return out_dir / job["group"] / f"{job['stem']}.json"
    return out_dir / job["group"] / job["cond"] / f"{job['stem']}.json"


def process_for_wer(tn, text: str, lang: str) -> str:
    text = tn.normalize(text, lang)
    if not text:
        return ""
    if lang == "zh":
        return " ".join(list(text.replace(" ", "")))
    return text.lower()


def process_for_cer(tn, text: str, lang: str) -> str:
    text = tn.normalize(text, lang)
    if not text:
        return ""
    if lang == "zh":
        return text.replace(" ", "")
    return text.lower().replace(" ", "")


def build_asr(device: str, whisper: str, paraformer: str, asr_py: str):
    """加载 ASR.py 里的逻辑，模型路径改到本机 / 服务器候选。"""
    asr_mod = load_module(asr_py, "moss_aba_asr_mod")
    import torch
    import zhconv
    from funasr import AutoModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    class QuietASR(asr_mod.ASR):
        def __init__(self, device="npu"):
            self.device = device
            self.en_processor = WhisperProcessor.from_pretrained(whisper)
            self.en_model = WhisperForConditionalGeneration.from_pretrained(whisper).to(self.device)
            self.en_model.float().eval()
            self.zh_model = AutoModel(model=paraformer, device=str(device), disable_update=True)

        @torch.no_grad()
        def asr(self, audio_path, lang="zh"):
            if lang == "en":
                # soundfile 读不了 mp3；wav/mp3 统一走 librosa
                import librosa
                import numpy as np

                wav, _ = librosa.load(str(audio_path), sr=16000, mono=True)
                wav = np.asarray(wav, dtype=np.float32)
                feats = self.en_processor(wav, sampling_rate=16000, return_tensors="pt").input_features
                feats = feats.to(self.device)
                forced = self.en_processor.get_decoder_prompt_ids(language="english", task="transcribe")
                ids = self.en_model.generate(feats, forced_decoder_ids=forced)
                return self.en_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if lang == "zh":
                # Paraformer / funasr 直接吃文件路径，wav 和 mp3 都可以
                res = self.zh_model.generate(input=audio_path, batch_size_s=300)
                text = res[0]["text"]
                return zhconv.convert(text, "zh-cn").strip()
            raise NotImplementedError(lang)

    return QuietASR(device=device)


def worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads", 2)))
    device = str(payload["device"])
    pin_npu(device)
    log_fp, log = open_worker_log(payload)
    n_ok = n_err = 0
    try:
        wer_mod = load_module(payload["wer_py"], f"moss_aba_wer_{payload.get('worker_id', 0)}")
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
                        "group": job["group"],
                        "cond": job["cond"],
                        "language": lang,
                        "asr_model": model_name,
                        "text_id": job.get("text_id"),
                        "ref": ref,
                        "hyp": hyp,
                        "wer": wer,
                        "cer": cer,
                    },
                )
                n_ok += 1
                if n_ok % 20 == 0:
                    log(f"ok={n_ok} {job['cond']} {job['stem']} {model_name} wer={wer} cer={cer}")
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
    p.add_argument(
        "--wav-root",
        type=str,
        default=str(EXP_ROOT / "moss_aba_emotion_wav_n1500"),
    )
    p.add_argument("--text-json", type=str, default=first_existing(*_TEXT_JSON))
    p.add_argument("--out_dir", type=str, default=str(EXP_ROOT / "moss_aba_asr_wer_n1500"))
    p.add_argument("--whisper", type=str, default=first_existing(*_WHISPER))
    p.add_argument("--paraformer", type=str, default=first_existing(*_PARAFORMER))
    p.add_argument("--asr-py", type=str, default=first_existing(*_ASR_PY))
    p.add_argument("--wer-py", type=str, default=first_existing(*_WER_PY))
    args = p.parse_args()

    for label, path in (
        ("text-json", args.text_json),
        ("asr-py", args.asr_py),
        ("wer-py", args.wer_py),
    ):
        if not Path(path).exists():
            raise SystemExit(f"missing {label}: {path}")

    refs = load_ref_text(Path(args.text_json))
    jobs = list_jobs(Path(args.wav_root), refs)
    if not jobs:
        raise SystemExit(f"no jobs under {args.wav_root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_zh = sum(1 for j in jobs if j["language"] == "zh")
    n_en = sum(1 for j in jobs if j["language"] == "en")
    print(
        f"jobs={len(jobs)} zh={n_zh} en={n_en}\n"
        f"  whisper={args.whisper}\n  paraformer={args.paraformer}\n"
        f"  text={args.text_json}\n  -> {out_dir}",
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
    dump_json(out_dir / "launch_summary.json", {"n_jobs": len(jobs), "n_zh": n_zh, "n_en": n_en, "workers": outs})
    print("done", outs, flush=True)


if __name__ == "__main__":
    main()
