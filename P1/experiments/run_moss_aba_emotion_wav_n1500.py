"""MOSS：指定层 ABA 替换，保存语音供情感 / 说话人分析。

数据：mix_1500。只做 MOSS。8 进程 × 8 NPU。

条件
----
1. 单层 ABA：L4, L5, L6, L7, L8, L9, L12, L18, L31。每一层用自己的 rare ABA。
2. 二层 ABA：上面 9 层的全部无序组合，C(9,2)=36。每一层用自己的 rare ABA。
3. L4–L14 写入时只用 L9 的同一对 ABA。
   - 全写：L4–L14 每一层都写，目录 ``band_L04_L14``
   - 单层保留原码：L4, L5, L6, L7, L8, L9, L12
   - 两层保留原码：这 7 层的全部两两组合
   跳过的层不写，其余层仍写 L9 的 ABA，目录 ``skip/``。

另存每句的干净重构 ``clean/{stem}.wav``。说话人和情感都相对原始语音计算。
已存在的 wav 跳过。

例::

  python -m P1.experiments.run_moss_aba_emotion_wav_n1500 \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from base.exp_parallel import launch_clip_workers, make_parser, resolve_out_dir
from base.io import match_len
from base.moss_io import decode_moss, encode_moss, read_wav, write_wav
from base.npu_parallel import ensure_spawn, free_device, limit_cpu_threads, open_worker_log, pin_npu
from base.rare_tokens import ensure_rare_table, layer_aba
from base.util import dump_json, load_model
from P1.experiments.run_codecs_band_layer_gap import aba_ids_np, list_mix_clips

SINGLE_LAYERS = (4, 5, 6, 7, 8, 9, 12, 18, 31)
BAND_LAYERS = tuple(range(4, 15))
DROP_POOL = (4, 5, 6, 7, 8, 9, 12)
OUT_STEM = "moss_aba_emotion_wav_n1500"


def rare_pair(table: dict, layer: int) -> tuple[int, int, str]:
    picked = layer_aba(table, int(layer))
    if picked:
        a, b = int(picked[0][0]), int(picked[0][1])
        if a != b:
            return a, b, f"rare_L{int(layer)}"
    picked9 = layer_aba(table, 9)
    if not picked9:
        raise RuntimeError(f"rare ABA missing for L{layer} and L9")
    a, b = int(picked9[0][0]), int(picked9[0][1])
    return a, b, f"rare_L9_fallback_for_L{int(layer)}"


def tag_layers(layers: list[int] | tuple[int, ...]) -> str:
    return "_".join(f"L{int(x):02d}" for x in layers)


def sample_drop_sets() -> list[list[int]]:
    """L4/L5/L6/L7/L8/L9/L12 的单层，以及全部两两组合。这些层保留原码。"""
    pool = list(DROP_POOL)
    sets = [[x] for x in pool]
    sets.extend([list(p) for p in itertools.combinations(pool, 2)])
    return sets


def apply_own_aba(codes: np.ndarray, layers: list[int], aba_by_layer: dict[int, tuple[int, int]]) -> np.ndarray:
    """单层、双层：每一层写入自己的 ABA。"""
    out = np.array(codes, dtype=np.int64, copy=True)
    t = int(out.shape[1])
    for layer in layers:
        a, b = aba_by_layer[int(layer)]
        out[int(layer)] = aba_ids_np(t, int(a), int(b))
    return out


def apply_l9_aba(codes: np.ndarray, layers: list[int], aba: tuple[int, int]) -> np.ndarray:
    """L4–L14：列出的层都写入 L9 的同一对 ABA。"""
    out = np.array(codes, dtype=np.int64, copy=True)
    t = int(out.shape[1])
    token = aba_ids_np(t, int(aba[0]), int(aba[1]))
    for layer in layers:
        out[int(layer)] = token
    return out


def decode_to(model, wav: np.ndarray, codes: np.ndarray, path: Path, sr: int) -> None:
    if path.exists():
        return
    wm = np.asarray(match_len(wav, decode_moss(model, codes)), dtype=np.float32).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, wm, int(sr))


def worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads", 2)))
    pin_npu(str(payload["device"]))
    log_fp, log = open_worker_log(payload)
    n_ok = n_err = 0
    try:
        model, sr = load_model(payload["model_path"], payload["device"])
        table = ensure_rare_table(payload.get("table") or payload.get("table_path"))
        a9, b9, aba_src = rare_pair(table, 9)
        aba_l9 = (int(a9), int(b9))
        own: dict[int, tuple[int, int]] = {}
        for layer in payload["singles"]:
            a, b, _src = rare_pair(table, int(layer))
            own[int(layer)] = (int(a), int(b))
        log(f"band ABA is L9 via {aba_src}: {aba_l9}")
        singles = [int(x) for x in payload["singles"]]
        pairs = [list(map(int, p)) for p in payload["pairs"]]
        band = [int(x) for x in payload["band"]]
        drops = [list(map(int, d)) for d in payload["drops"]]
        out_dir = Path(payload["out_dir"])
        for stem, path in payload["items"]:
            try:
                wav = read_wav(path, sr)
                codes = np.asarray(encode_moss(model, wav), dtype=np.int64)
                decode_to(model, wav, codes, out_dir / "clean" / f"{stem}.wav", sr)
                for layer in singles:
                    decode_to(
                        model,
                        wav,
                        apply_own_aba(codes, [layer], own),
                        out_dir / "single" / tag_layers([layer]) / f"{stem}.wav",
                        sr,
                    )
                for pair in pairs:
                    decode_to(
                        model,
                        wav,
                        apply_own_aba(codes, pair, own),
                        out_dir / "pair" / tag_layers(pair) / f"{stem}.wav",
                        sr,
                    )
                band_codes = apply_l9_aba(codes, band, aba_l9)
                decode_to(model, wav, band_codes, out_dir / "band_L04_L14" / f"{stem}.wav", sr)
                for skip in drops:
                    kept = [layer for layer in band if layer not in skip]
                    decode_to(
                        model,
                        wav,
                        apply_l9_aba(codes, kept, aba_l9),
                        out_dir / "skip" / tag_layers(skip) / f"{stem}.wav",
                        sr,
                    )
                n_ok += 1
                if n_ok % 10 == 0:
                    log(f"ok={n_ok} {stem}")
            except Exception as exc:
                n_err += 1
                log(f"ERR {stem}: {exc}")
        free_device()
    finally:
        if log_fp is not None:
            log_fp.close()
    return {"worker_id": payload.get("worker_id"), "n_ok": n_ok, "n_error": n_err}


def main() -> None:
    ensure_spawn()
    p = make_parser(__doc__, default_out="", with_names=False)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    out_dir = resolve_out_dir(args, OUT_STEM)
    clips = list_mix_clips(Path(args.wav_dir), int(args.limit))
    if not clips:
        raise SystemExit(f"no audio in {args.wav_dir}")
    singles = list(SINGLE_LAYERS)
    pairs = [list(p) for p in itertools.combinations(singles, 2)]
    drops = sample_drop_sets()
    dump_json(
        out_dir / "conditions.json",
        {
            "codec": "moss",
            "singles": singles,
            "pairs": pairs,
            "band": list(BAND_LAYERS),
            "drop_pool": list(DROP_POOL),
            "drops": drops,
            "aba_single_pair": "each layer uses its own rare ABA",
            "aba_band": "L9 rare token on every written layer",
            "aba_layer": 9,
        },
    )
    print(
        f"clips={len(clips)} singles={len(singles)} pairs={len(pairs)} drops={len(drops)} -> {out_dir}",
        flush=True,
    )
    outs = launch_clip_workers(
        worker,
        clips,
        args,
        out_dir=out_dir,
        common={
            "out_dir": str(out_dir),
            "model_path": args.model_path,
            "table": args.table,
            "singles": singles,
            "pairs": pairs,
            "band": list(BAND_LAYERS),
            "drops": drops,
        },
    )
    dump_json(out_dir / "launch_summary.json", {"n_clips": len(clips), "workers": outs})
    print("done", outs, flush=True)


if __name__ == "__main__":
    main()
