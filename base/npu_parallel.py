"""8 进程 × 8 NPU 分片运行（与 codec_an eval/audioseal 规模脚本同协议）。

默认 ``--num-workers 8 --npu-ids 0,1,2,3,4,5,6,7``。单卡调试用
``--num-workers 1 --device npu:0``（或 cuda:0）。
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


def parse_npu_ids(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def add_npu_args(parser, *, default_workers: int = 8, default_device: str = "npu:0"):
    """给 argparse 加上多 NPU 参数。"""
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="仅 --num-workers 1 时使用；多 worker 时忽略，改用 --npu-ids",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(default_workers),
        help=">1 时按任务分片到多 NPU（默认 8）",
    )
    parser.add_argument("--npu-ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser


def limit_cpu_threads(n: int = 1) -> None:
    n = max(1, int(n))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        pass


def pin_npu(device: str) -> None:
    """绑定当前进程到指定 NPU；非 npu 设备则跳过。"""
    if not str(device).startswith("npu"):
        return
    import torch
    import torch_npu  # noqa: F401

    if ":" in str(device):
        torch.npu.set_device(int(str(device).split(":", 1)[1]))


def free_device() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                torch.npu.empty_cache()
        except Exception:
            pass


def shard_round_robin(items: Sequence[Any], n_shards: int) -> list[list[Any]]:
    n = max(1, int(n_shards))
    shards: list[list[Any]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        shards[i % n].append(item)
    return shards


def build_payloads(
    items: Sequence[Any],
    *,
    num_workers: int,
    npu_ids: Sequence[int],
    device_single: str,
    logs_dir: Path | None = None,
    common: dict | None = None,
    items_key: str = "items",
) -> list[dict]:
    """把 items 分片成 worker payload 列表。"""
    common = dict(common or {})
    n_workers = max(1, int(num_workers))
    if n_workers == 1:
        pl = {
            "worker_id": 0,
            "device": str(device_single),
            items_key: list(items),
            **common,
        }
        if logs_dir is not None:
            pl["worker_log"] = str(Path(logs_dir) / "w0.log")
        return [pl]

    if len(npu_ids) < n_workers:
        raise SystemExit(f"--num-workers={n_workers} needs >= {n_workers} --npu-ids")
    shards = shard_round_robin(items, n_workers)
    payloads = []
    for wid, shard in enumerate(shards):
        pl = {
            "worker_id": wid,
            "device": f"npu:{int(npu_ids[wid])}",
            items_key: list(shard),
            **common,
        }
        if logs_dir is not None:
            pl["worker_log"] = str(Path(logs_dir) / f"w{wid}.log")
        payloads.append(pl)
        print(
            f"  worker {wid} -> npu:{npu_ids[wid]} {items_key}={len(shard)}",
            flush=True,
        )
    return payloads


def run_shards(
    worker_fn: Callable[[dict], dict],
    payloads: Sequence[dict],
    *,
    num_workers: int,
) -> list[dict]:
    """顺序或 ProcessPool 跑分片，返回各 worker 的结果 dict 列表（按完成顺序）。"""
    n_workers = max(1, int(num_workers))
    if n_workers == 1 or len(payloads) == 1:
        return [worker_fn(payloads[0])]

    outs: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(worker_fn, pl): pl["worker_id"] for pl in payloads}
        for fut in as_completed(futs):
            wid = futs[fut]
            out = fut.result()
            print(
                f"  worker {wid} done n={out.get('n_ok', out.get('n_files', '?'))} "
                f"err={out.get('n_error', 0)}",
                flush=True,
            )
            outs.append(out)
    return outs


def open_worker_log(payload: dict):
    """返回 (log_fp or None, log_fn)。"""
    log_path = payload.get("worker_log")
    worker_id = int(payload.get("worker_id", 0))
    log_fp = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(log_path, "a", encoding="utf-8")

    def _log(msg: str) -> None:
        line = f"[w{worker_id}] {msg}"
        print(line, flush=True)
        if log_fp is not None:
            log_fp.write(line + "\n")
            log_fp.flush()

    return log_fp, _log


def ensure_spawn() -> None:
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
