"""Latent-Mark 抗 codec：MOSS / Qwen round-trip 后检测是否仍存活。

流程::

  1. 用 clean 集标定 Latent-Mark（默认 snac_24 + Latent-Cluster）
  2. 对测试句 embed → 检测（无攻击）
  3. 对带水印波形做 MOSS / Qwen encode→decode → 再检测
  4. 汇总 TPR / 存活率（score > 0）

例::

  python -m P1.experiments.run_latentmark_vs_moss_qwen \\
    --wav-dir /path/mix_1500 \\
    --n-calib 40 --n-test 50 \\
    --codecs moss,qwen \\
    --device cuda \\
    --out-dir /path/latentmark_vs_moss_qwen
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from base.config import EXP_ROOT, default_mix_1500, default_model_path, first_existing
from base.util import dump_json
from P1.latentmark_api import LatentMarkAPI, codec_roundtrip_attack, list_audio
from P2.experiments.token_transition.backends import DAC_PATH_DEFAULT, QWEN_PATH_DEFAULT
from P2.experiments.token_transition.common import fmt

OUT_STEM = "latentmark_vs_moss_qwen"
_OUT_CANDS = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
    Path(r"D:\KimiData\kimi\Workspaces\P1\exp\P1") / OUT_STEM,
    Path(r"E:\code\water_marks\result") / OUT_STEM,
    EXP_ROOT / OUT_STEM,
)
DEFAULT_OUT = next((p for p in _OUT_CANDS if p.parent.exists()), _OUT_CANDS[-1])


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_wav(path: Path, wav: np.ndarray | torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    sf.write(str(path), x, int(sr))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav-dir", type=str, default=default_mix_1500(), help="干净语音目录（标定+测试）")
    p.add_argument("--n-calib", type=int, default=40)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--seconds", type=float, default=5.0, help="裁到前 N 秒；0=全长")
    p.add_argument("--codecs", type=str, default="moss,qwen", help="攻击 codec，CSV")
    p.add_argument("--views", type=str, default="snac_24", help="Latent-Mark surrogate views")
    p.add_argument("--axis", type=str, default="cluster", choices=("cluster", "pca", "random"))
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--latentmark-src", type=str, default="")
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--dac-path", type=str, default=str(DAC_PATH_DEFAULT or ""))
    p.add_argument(
        "--qwen-path",
        type=str,
        default=str(
            QWEN_PATH_DEFAULT
            or first_existing(Path(r"E:\pred_model\Qwen3-TTS-12Hz-1.7B-Base\speech_tokenizer"), "")
        ),
    )
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-wav", action="store_true", default=True)
    p.add_argument("--no-save-wav", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_save_wav:
        args.save_wav = False
    if args.no_resume:
        args.resume = False

    rng = np.random.default_rng(int(args.seed))
    files = list_audio(args.wav_dir, recursive=True)
    if not files:
        raise SystemExit(f"no audio in {args.wav_dir}")
    rng.shuffle(files)
    n_calib = int(args.n_calib)
    n_test = int(args.n_test)
    need = n_calib + n_test
    if len(files) < need:
        raise SystemExit(f"need>={need} files, got {len(files)}")
    calib_files = files[:n_calib]
    test_files = files[n_calib:need]
    codecs = [c.strip().lower() for c in args.codecs.split(",") if c.strip()]
    for c in codecs:
        if c not in ("moss", "qwen", "dac"):
            raise SystemExit(f"unsupported attack codec {c}; use moss,qwen,dac")
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    out = Path(args.out_dir)
    for d in ("per_clip", "wav", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)

    print(
        f"Latent-Mark views={views} axis={args.axis} device={args.device}\n"
        f"calib={len(calib_files)} test={len(test_files)} attacks={codecs}",
        flush=True,
    )

    api = LatentMarkAPI(
        views=views,
        device=str(args.device),
        axis=str(args.axis),
        steps=int(args.steps),
        seed=int(args.seed),
        latentmark_src=args.latentmark_src or None,
    )
    state_path = out / "latentmark_state.pt"
    if args.resume and state_path.is_file():
        print(f"[calib] load {state_path}", flush=True)
        api = LatentMarkAPI.load(state_path, device=str(args.device), latentmark_src=args.latentmark_src or None)
    else:
        print("[calib] fitting null distribution …", flush=True)
        calib_info = api.calibrate(calib_files, seconds=float(args.seconds) if args.seconds > 0 else 5.0)
        api.save(state_path)
        dump_json(out / "calib.json", calib_info)

    thr = LatentMarkAPI.THRESHOLD
    rows = []
    t0 = time.time()
    sec = float(args.seconds) if float(args.seconds) > 0 else None

    for i, path in enumerate(test_files):
        stem = Path(path).stem
        dest = out / "per_clip" / f"{stem}.json"
        if args.resume and dest.is_file():
            try:
                rows.append(json.loads(dest.read_text(encoding="utf-8")))
                continue
            except Exception:
                pass
        try:
            wav_c, sr_c = api.lm.load_audio(path, seconds=sec)
            score_clean = api.detect(wav_c, sr_c)

            wav_wm, sr_wm, emb_info = api.embed(wav_c, sr_c)
            score_wm = float(emb_info.get("wm_score", api.detect(wav_wm, sr_wm)))

            if args.save_wav:
                save_wav(out / "wav" / f"{stem}__clean.wav", wav_c, sr_c)
                save_wav(out / "wav" / f"{stem}__wm.wav", wav_wm, sr_wm)

            atk = {}
            for codec in codecs:
                y, sr_y = codec_roundtrip_attack(
                    wav_wm,
                    sr_wm,
                    codec=codec,
                    device=str(args.device),
                    moss_path=str(args.moss_path),
                    dac_path=str(args.dac_path),
                    qwen_path=str(args.qwen_path),
                )
                # detect at attack output sr
                sc = api.detect(y, sr_y)
                atk[codec] = {
                    "score": float(sc),
                    "detected": bool(sc > thr),
                    "sr": int(sr_y),
                }
                if args.save_wav:
                    save_wav(out / "wav" / f"{stem}__atk_{codec}.wav", y, sr_y)

                # clean through same attack (for delta-score / transferability)
                y0, sr0 = codec_roundtrip_attack(
                    wav_c,
                    sr_c,
                    codec=codec,
                    device=str(args.device),
                    moss_path=str(args.moss_path),
                    dac_path=str(args.dac_path),
                    qwen_path=str(args.qwen_path),
                )
                sc0 = api.detect(y0, sr0)
                atk[codec]["clean_attacked_score"] = float(sc0)
                atk[codec]["delta_score"] = float(sc - sc0)

            row = {
                "name": stem,
                "src": path,
                "score_clean": float(score_clean),
                "score_wm": float(score_wm),
                "detected_clean": bool(score_clean > thr),
                "detected_wm": bool(score_wm > thr),
                "attacks": atk,
                "sr_wm": int(sr_wm),
            }
            rows.append(row)
            _write_json(dest, row)
            msg = f"[{i+1}/{len(test_files)}] {stem}: clean={fmt(score_clean)} wm={fmt(score_wm)}"
            for c, a in atk.items():
                msg += f" | {c}={fmt(a['score'])}({'Y' if a['detected'] else 'N'})"
            print(msg, flush=True)
        except Exception as e:
            err = {"name": stem, "src": path, "error": str(e), "trace": traceback.format_exc()}
            _write_json(out / "per_clip" / f"{stem}__ERR.json", err)
            print(f"ERR {stem}: {e}", flush=True)

    # summary
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    summary = {
        "n_test": n,
        "n_calib": len(calib_files),
        "threshold": thr,
        "method": api.name,
        "views": views,
        "axis": args.axis,
        "device": args.device,
        "detectability": {
            "tpr": float(np.mean([r["detected_wm"] for r in ok])) if n else float("nan"),
            "fpr": float(np.mean([r["detected_clean"] for r in ok])) if n else float("nan"),
            "mean_score_wm": float(np.mean([r["score_wm"] for r in ok])) if n else float("nan"),
            "mean_score_clean": float(np.mean([r["score_clean"] for r in ok])) if n else float("nan"),
        },
        "survivability": {},
        "transferability": {},
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    for codec in codecs:
        det = [r["attacks"][codec]["detected"] for r in ok if codec in r.get("attacks", {})]
        scs = [r["attacks"][codec]["score"] for r in ok if codec in r.get("attacks", {})]
        deltas = [
            r["attacks"][codec]["delta_score"]
            for r in ok
            if codec in r.get("attacks", {}) and "delta_score" in r["attacks"][codec]
        ]
        summary["survivability"][codec] = {
            "tpr": float(np.mean(det)) if det else float("nan"),
            "mean_score": float(np.mean(scs)) if scs else float("nan"),
            "n": len(det),
        }
        summary["transferability"][codec] = {
            "pos_delta_rate": float(np.mean([d > 0 for d in deltas])) if deltas else float("nan"),
            "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
        }

    dump_json(out / "summary.json", {"summary": summary, "rows": ok})
    lines = [
        "# Latent-Mark vs MOSS / Qwen codec round-trip\n\n",
        f"- method: `{api.name}` views=`{views}` axis=`{args.axis}`\n",
        f"- n_calib={len(calib_files)} n_test={n} threshold={thr}\n\n",
        "## Detectability (no attack)\n\n",
        f"- TPR(wm)={summary['detectability']['tpr']:.4f}\n",
        f"- FPR(clean)={summary['detectability']['fpr']:.4f}\n",
        f"- mean score wm/clean = {summary['detectability']['mean_score_wm']:.4f} / "
        f"{summary['detectability']['mean_score_clean']:.4f}\n\n",
        "## Survivability after codec attack\n\n",
        "| codec | TPR | mean_score | pos_delta_rate |\n|---|---:|---:|---:|\n",
    ]
    for codec in codecs:
        s = summary["survivability"][codec]
        t = summary["transferability"][codec]
        lines.append(
            f"| {codec} | {s['tpr']:.4f} | {s['mean_score']:.4f} | {t['pos_delta_rate']:.4f} |\n"
        )
    (out / "FINDINGS.md").write_text("".join(lines), encoding="utf-8")
    print(f"DONE -> {out}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
