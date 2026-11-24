"""通用工具：模型加载包装、JSON、列目录。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import EXP_ROOT, default_model_path, default_table_path
from .moss_io import SEED, decode_moss, encode_moss, load_moss, set_seed
from .plan import apply_plan, build_rare_cycle_plan
from .rare_tokens import ensure_rare_table
from .wave import _pad_or_trim

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".mp4", ".m4a"}


def load_model(model_path: str, device: str):
    return load_moss(model_path, device)


def auto_device() -> str:
    import torch

    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu:0"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def add_model_args(p, *, n_files: int = 3):
    from .npu_parallel import add_npu_args

    p.add_argument("--model_path", type=str, default=default_model_path())
    p.add_argument("--wav-dir", type=str, default=str(Path(r"e:\data\Clotho\aishell1\test\wav")))
    p.add_argument("--n-files", "--n_files", type=int, default=int(n_files))
    p.add_argument("--high-start", type=int, default=9)
    p.add_argument("--high-end", type=int, default=9)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--table", type=str, default=default_table_path())
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--no-seed", action="store_true")
    p.add_argument("--save-wav", action="store_true")
    # 默认 8 worker × npu:0–7；--device 仅在 num-workers=1 时生效
    add_npu_args(p, default_workers=8, default_device="npu:0")
    return p


def prepare_run(args, stem: str):
    import os

    from .npu_parallel import parse_npu_ids

    if getattr(args, "no_seed", False):
        os.environ["WATERMARK_NO_SEED"] = "1"
        seed = None
    else:
        set_seed(SEED)
        seed = SEED
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", "") else (EXP_ROOT / stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = ensure_rare_table(path=getattr(args, "table", default_table_path()))
    device = str(getattr(args, "device", "") or "").strip() or auto_device()
    he = int(getattr(args, "high_end", 9))
    n_workers = max(1, int(getattr(args, "num_workers", 1) or 1))
    npu_ids = parse_npu_ids(getattr(args, "npu_ids", "0") or "0")
    return {
        "seed": seed,
        "out_dir": out_dir,
        "table": table,
        "device": device,
        "high_start": int(getattr(args, "high_start", 9)),
        "high_end": None if he < 0 else he,
        "threshold": float(getattr(args, "threshold", 0.5)),
        "num_workers": n_workers,
        "npu_ids": npu_ids,
        "cpu_threads": int(getattr(args, "cpu_threads", 1) or 1),
    }


def dump_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
    return path


def list_audio(wav_dir, n_files: int | None = None) -> list[Path]:
    d = Path(wav_dir)
    files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if n_files is not None and int(n_files) > 0:
        files = files[: int(n_files)]
    return files


def render_watermarked_wav(model, wav, tokens, plan):
    mod, n_changed = apply_plan(tokens, plan)
    wm_full = decode_moss(model, mod)
    n = int(np.asarray(wav).reshape(-1).shape[0])
    wm_full = np.asarray(_pad_or_trim(wm_full, n), dtype=np.float32).reshape(-1)
    return wm_full, mod, n_changed, wm_full


def embed_wav(
    model,
    wav: np.ndarray,
    table: dict,
    high_start: int,
    high_end: int | None,
    layers=None,
    period: int = 3,
):
    tokens = encode_moss(model, wav)
    n_q, T = tokens.shape
    plan = build_rare_cycle_plan(
        T,
        n_q=n_q,
        high_start=int(high_start),
        high_end=high_end,
        table=table,
        layers=layers,
        period=int(period),
    )
    wm, mod, n_changed, wm_full = render_watermarked_wav(model, wav, tokens, plan)
    return {
        "wav": np.asarray(wm, dtype=np.float32).reshape(-1),
        "tokens": tokens,
        "mod": mod,
        "plan": plan,
        "n_changed": int(n_changed),
        "wm_full": wm_full,
        "T": int(T),
        "n_q": int(n_q),
        "period": int(period),
    }
