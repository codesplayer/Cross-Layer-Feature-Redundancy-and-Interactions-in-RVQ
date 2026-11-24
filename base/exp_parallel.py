"""实验脚本统一 8×NPU 启动脚手架。

典型用法（按 clip 分片）::

    def _worker(payload: dict) -> dict:
        ...
        return {"worker_id": ..., "rows": [...], "n_ok": ..., "n_error": 0}

    def main():
        args = parse_args()  # 含 add_npu_args
        outs = launch_clip_workers(_worker, clips, args, out_dir=out_dir, common={...})
        rows = flatten_key(outs, "rows")
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import (
    DEFAULT_WAV_DIR,
    EXP_ROOT,
    NAMES,
    default_mix_1500,
    default_model_path,
    default_table_path,
)
from .npu_parallel import (
    add_npu_args,
    build_payloads,
    ensure_spawn,
    parse_npu_ids,
    run_shards,
)


def make_parser(description: str, *, default_out: str = "", with_names: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model_path", type=str, default=default_model_path())
    # Prefer local/server mix_1500 when present
    try:
        _wav_default = default_mix_1500()
    except Exception:
        _wav_default = str(DEFAULT_WAV_DIR)
    p.add_argument("--wav-dir", type=str, default=_wav_default)
    p.add_argument("--table", type=str, default=default_table_path())
    p.add_argument("--out_dir", type=str, default=default_out)
    if with_names:
        p.add_argument("--names", type=str, default=",".join(NAMES))
    p.add_argument("--no-sv", action="store_true")
    add_npu_args(p, default_workers=8, default_device="npu:0")
    return p


def resolve_out_dir(args, stem: str) -> Path:
    out = Path(args.out_dir) if getattr(args, "out_dir", "") else (EXP_ROOT / stem)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def parse_names_arg(args) -> list[str]:
    return [x.strip() for x in str(getattr(args, "names", "") or "").split(",") if x.strip()]


def resolve_named_clips(wav_dir: Path | str, names: Sequence[str]) -> list[tuple[str, str]]:
    root = Path(wav_dir)
    clips = []
    for name in names:
        stem = Path(name).stem
        p = root / f"{stem}.wav"
        if not p.exists():
            raise SystemExit(f"missing wav: {p}")
        clips.append((stem, str(p)))
    return clips


def launch_clip_workers(
    worker_fn: Callable[[dict], dict],
    items: Sequence[Any],
    args,
    *,
    out_dir: Path,
    common: dict | None = None,
    items_key: str = "items",
) -> list[dict]:
    """按 items 分片到 num_workers×npu，返回 worker 结果列表。"""
    n_workers = max(1, int(getattr(args, "num_workers", 8) or 8))
    npu_ids = parse_npu_ids(getattr(args, "npu_ids", "0,1,2,3,4,5,6,7"))
    logs_dir = Path(out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    base_common = {
        "model_path": getattr(args, "model_path", default_model_path()),
        "table_path": getattr(args, "table", default_table_path()),
        "wav_root": str(getattr(args, "wav_dir", DEFAULT_WAV_DIR)),
        "wav_out": str(Path(out_dir) / "wav"),
        "cpu_threads": int(getattr(args, "cpu_threads", 1) or 1),
        "no_sv": bool(getattr(args, "no_sv", False)),
        "dac_path": str(getattr(args, "dac_path", "") or ""),
        "tokenizer_path": str(getattr(args, "tokenizer_path", "") or ""),
    }
    if common:
        base_common.update(common)
    print(f"launch workers={n_workers} items={len(items)} out={out_dir}", flush=True)
    payloads = build_payloads(
        items,
        num_workers=n_workers,
        npu_ids=npu_ids,
        device_single=str(getattr(args, "device", "npu:0")),
        logs_dir=logs_dir,
        common=base_common,
        items_key=items_key,
    )
    return run_shards(worker_fn, payloads, num_workers=n_workers)


def flatten_key(outs: Sequence[dict], key: str) -> list:
    rows: list = []
    for o in outs:
        rows.extend(o.get(key) or [])
    return rows


def merge_dict_lists(outs: Sequence[dict], key: str) -> dict:
    acc: dict = {}
    for o in outs:
        for k, vv in (o.get(key) or {}).items():
            acc.setdefault(k, []).extend(list(vv))
    return acc
