"""MOSS / Qwen3 / DAC：二层组合 band ABA 水印 gap 实验。

协议
----
1. 各 codec 在固定层池内枚举 **全部无序二层组合** C(n,2)：
   - MOSS：L0–14 → C(15,2)=105
   - Qwen：L0–8 → C(9,2)=36
   - DAC：L0–10 → C(11,2)=55
2. encode → **同时替换这两层** 为各自 rare ABA（与单层实验相同 token）→ decode
3. 检测（band）：各写入层用**该层** ABA 模板算 cos（L0↔L0，L1↔L1，…）
4. 总 gap：``gapA_sum = Σ cos(wm,tmpl_L) − Σ cos(orig,tmpl_L)``
5. 另报 ``gapA_write``（写入码字、未再编码）对照是否嵌印成功
6. 质量：wave corr / LSD / MCD（相对干净原波形）——与单层实验相同

每句结果：``{out}/{codec}/per_clip/{stem}.json``（``rows`` 含该句所有层对）

加速
----
- **句间**：``--num-workers`` × NPU 分片（已有）
- **句内**：原音频 encode / 各层 ABA 模板 / orig z_q 只算一次，多 pair 复用；
  每个 pair 仍需各自 decode + 再编码 wm（写入层不同，波形不同）

例::

  $env:PYTHONPATH = "e:\\code\\water_marks"
  python -m P1.experiments.run_codecs_band_two_layer_random \\
    --codecs moss,qwen,dac --limit 3 \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from base.config import EXP_ROOT, default_mix_1500, default_table_path
from base.exp_parallel import flatten_key, launch_clip_workers, make_parser, resolve_out_dir
from base.io import match_len
from base.moss_io import decode_moss, encode_moss, read_wav, set_seed
from base.npu_parallel import ensure_spawn, free_device, limit_cpu_threads, open_worker_log, pin_npu
from base.rare_tokens import ensure_rare_table
from base.util import dump_json

from base.residual_feat import codes_to_zq_sum

from P1.experiments.run_codecs_band_layer_gap import (
    DEFAULT_CODECS,
    _fmt,
    _make_quality_tools,
    _mean,
    _wave_corr,
    aba_ids_np,
    aba_ids_torch,
    band_tmpl_dac,
    band_tmpl_moss,
    band_tmpl_qwen,
    dac_decode,
    dac_encode,
    dac_layer_zq,
    decode_codes_kt,
    default_moss_path,
    default_qwen_tokenizer,
    detect_gapA_band,
    encode_codes_kt,
    list_mix_clips,
    load_codec,
    n_q_of,
    qwen_layer_zq,
    resolve_aba_for_layer,
    token_agree_layer,
    wave_mcd_lsd,
)

OUT_STEM = "codecs_band_two_layer_random"
_LOCAL_EXP = Path(r"/data1/home/ynian/codes/eval/add_exp/result")
DEFAULT_OUT = (EXP_ROOT if EXP_ROOT.exists() else _LOCAL_EXP) / OUT_STEM

# 各 codec 二层随机的层池（含端点）
LAYER_POOL = {
    "moss": list(range(0, 32)),  # L0–14
    "qwen": list(range(0, 16)),  # L0–8
    "dac": list(range(0, 32)),  # L0–10
}


def pool_for(codec: str, handle) -> list[int]:
    codec = str(codec).lower()
    n_q = n_q_of(codec, handle)
    pool = [int(L) for L in LAYER_POOL[codec] if 0 <= int(L) < n_q]
    if len(pool) < 2:
        raise ValueError(f"{codec}: layer pool too small after n_q={n_q}: {pool}")
    return pool


def all_layer_pairs(pool: list[int]) -> list[list[int]]:
    """层池内全部无序二层组合，按 (L_i, L_j) 升序。"""
    return [[int(a), int(b)] for a, b in itertools.combinations(sorted(int(x) for x in pool), 2)]


def embed_two_layers(
    codec: str,
    handle,
    wav: np.ndarray,
    sr: int,
    layers: list[int],
    aba_by_L: dict[int, tuple[int, int, str]],
    *,
    codes_base=None,
) -> dict:
    """同时替换多个层，各层用各自 ABA。

    ``codes_base`` 若给定则不再 encode 原音频（同句多 pair 复用）。
    """
    codec = str(codec).lower()
    layers = [int(L) for L in layers]

    if codec == "moss":
        codes = (
            np.asarray(codes_base, dtype=np.int64).copy()
            if codes_base is not None
            else np.asarray(encode_moss(handle, wav), dtype=np.int64)
        )
        n_q, T = codes.shape
        codes = codes.copy()
        n_changed = 0
        for L in layers:
            a, b, _ = aba_by_L[L]
            before = codes[L].copy()
            codes[L] = aba_ids_np(T, a, b)
            n_changed += int(np.sum(codes[L] != before))
        wm = np.asarray(match_len(wav, decode_moss(handle, codes)), dtype=np.float32).reshape(-1)
        return {
            "wav": wm,
            "codes": codes,
            "T": int(T),
            "n_q": int(n_q),
            "n_changed": n_changed,
            "out_sr": int(sr),
        }

    if codec == "qwen":
        codes = codes_base.clone() if codes_base is not None else encode_codes_kt(handle, wav, sr)
        K, T = codes.shape
        codes = codes.clone()
        n_changed = 0
        for L in layers:
            a, b, _ = aba_by_L[L]
            before = codes[L].detach().clone()
            codes[L] = aba_ids_torch(T, a, b, device=codes.device)
            n_changed += int((codes[L] != before).sum().item())
        wm, out_sr = decode_codes_kt(handle, codes)
        wm = np.asarray(match_len(wav, wm), dtype=np.float32).reshape(-1)
        return {
            "wav": wm,
            "codes": codes,
            "T": int(T),
            "n_q": int(K),
            "n_changed": n_changed,
            "out_sr": int(out_sr),
        }

    codes = codes_base.clone() if codes_base is not None else dac_encode(handle, wav, sr)
    n, T = codes.shape
    codes = codes.clone()
    n_changed = 0
    for L in layers:
        a, b, _ = aba_by_L[L]
        before = codes[L].clone()
        codes[L] = aba_ids_torch(T, a, b, device=codes.device)
        n_changed += int((codes[L] != before).sum().item())
    wm = np.asarray(match_len(wav, dac_decode(handle, codes)), dtype=np.float32).reshape(-1)
    return {
        "wav": wm,
        "codes": codes,
        "T": int(T),
        "n_q": int(n),
        "n_changed": n_changed,
        "out_sr": int(handle.sample_rate),
    }


def band_tmpl_layer(codec, handle, T: int, layer: int, a: int, b: int) -> np.ndarray:
    codec = str(codec).lower()
    if codec == "moss":
        return band_tmpl_moss(handle, T, layer, a, b)
    if codec == "qwen":
        return band_tmpl_qwen(handle, T, layer, a, b)
    return band_tmpl_dac(handle, T, layer, a, b)


def layer_zq_from_codes(codec, handle, codes, layer: int) -> np.ndarray:
    """从已有 codes 取**该层** z_q（不重新 encode）。"""
    codec = str(codec).lower()
    L = int(layer)
    if codec == "moss":
        c = np.asarray(codes, dtype=np.int64)
        return codes_to_zq_sum(handle, c, L, L, layers=[L])
    if codec == "qwen":
        return qwen_layer_zq(handle, codes[L], L)
    return dac_layer_zq(handle, codes, L)


def reencode_codes(codec, handle, wav, sr):
    codec = str(codec).lower()
    if codec == "moss":
        return np.asarray(encode_moss(handle, wav), dtype=np.int64)
    if codec == "qwen":
        return encode_codes_kt(handle, wav, sr)
    return dac_encode(handle, wav, sr)


def process_clip_two_layer(
    codec: str,
    handle,
    wav: np.ndarray,
    *,
    sr: int,
    pairs: list[list[int]],
    table,
    aba_mode: str,
    a_cli: int,
    b_cli: int,
    token_seed: int,
    tmp_dir: Path,
    clip_stem: str,
    lsd_tool=None,
    mcd_tool=None,
) -> list[dict]:
    """对 ``pairs`` 中每一对层各做一次二层嵌入与检测。

    同句内：原音频只 encode 一次；ABA 与模板按层缓存；各 pair 只做
    replace→decode→再编码 wm（这是必然的，因写入层不同）。
    """
    codec = str(codec).lower()
    # 原音频 encode 一次，供全部 pair 复用
    codes_o = reencode_codes(codec, handle, wav, sr)
    T0 = int(codes_o.shape[-1])

    used_layers = sorted({int(L) for pair in pairs for L in pair})
    aba_cache: dict[int, tuple[int, int, str]] = {}
    for L in used_layers:
        aba_cache[L] = resolve_aba_for_layer(
            codec=codec,
            layer=L,
            table=table,
            aba_mode=aba_mode,
            a_cli=a_cli,
            b_cli=b_cli,
            token_seed=token_seed,
        )
    # 模板只依赖 (L,a,b,T)；同句 T 固定，按层缓存
    tmpl_cache: dict[int, np.ndarray] = {}
    for L in used_layers:
        a, b, _ = aba_cache[L]
        tmpl_cache[L] = band_tmpl_layer(codec, handle, T0, L, a, b)

    # orig 各层 z_q 也只算一次
    feat_o_cache: dict[int, np.ndarray] = {
        L: layer_zq_from_codes(codec, handle, codes_o, L) for L in used_layers
    }

    rows = []
    for pair_i, write_layers in enumerate(pairs):
        write_layers = [int(write_layers[0]), int(write_layers[1])]
        aba_by_L = {L: aba_cache[L] for L in write_layers}

        emb = embed_two_layers(
            codec, handle, wav, sr, write_layers, aba_by_L, codes_base=codes_o
        )
        wm = emb["wav"]
        codes_write = emb["codes"]
        out_sr = int(emb["out_sr"] or sr)
        det_sr_w = out_sr if codec == "qwen" else sr

        codes_w = reencode_codes(codec, handle, wm, det_sr_w)

        per_layer = []
        for L in write_layers:
            a, b, src = aba_by_L[L]
            feat_o = feat_o_cache[L]
            feat_w = layer_zq_from_codes(codec, handle, codes_w, L)
            feat_write = layer_zq_from_codes(codec, handle, codes_write, L)
            tmpl = tmpl_cache[L]
            # 再编码后 T 可能略变；模板按实际 T 对齐截断/重算
            T = int(max(feat_o.shape[-1], feat_w.shape[-1], emb["T"]))
            if int(tmpl.shape[-1]) != T:
                tmpl = band_tmpl_layer(codec, handle, T, L, a, b)
            det = detect_gapA_band(feat_o, feat_w, tmpl)
            det_write = detect_gapA_band(feat_o, feat_write, tmpl)
            agree = token_agree_layer(codes_w, L, a, b)
            agree_write = token_agree_layer(codes_write, L, a, b)
            per_layer.append(
                {
                    "write_layer": int(L),
                    "aba": [int(a), int(b), int(a)],
                    "aba_source": src,
                    "orig_cosine": det["orig_cosine"],
                    "wm_cosine": det["wm_cosine"],
                    "gapA": det["gapA"],
                    "write_cosine": float(det_write["wm_cosine"]),
                    "gapA_write": float(det_write["gapA"]),
                    "token_agree_wm": float(agree),
                    "token_agree_write": float(agree_write),
                }
            )

        q = wave_mcd_lsd(
            wav,
            wm,
            int(sr),
            tmp_dir=tmp_dir,
            stem=f"{clip_stem}_L{write_layers[0]}_{write_layers[1]}",
            lsd_tool=lsd_tool,
            mcd_tool=mcd_tool,
        )
        # 总 gap = Σ_L cos(wm, tmpl_L) − Σ_L cos(orig, tmpl_L)
        wm_cos_sum = float(sum(x["wm_cosine"] for x in per_layer))
        orig_cos_sum = float(sum(x["orig_cosine"] for x in per_layer))
        gap_sum = float(wm_cos_sum - orig_cos_sum)
        write_cos_sum = float(sum(x["write_cosine"] for x in per_layer))
        gap_write_sum = float(write_cos_sum - orig_cos_sum)
        rows.append(
            {
                "name": clip_stem,
                "codec": codec,
                "pair_index": int(pair_i),
                "write_layers": write_layers,
                "pair_key": f"L{write_layers[0]}_L{write_layers[1]}",
                "aba_by_layer": {
                    f"L{L}": {
                        "aba": [aba_by_L[L][0], aba_by_L[L][1], aba_by_L[L][0]],
                        "aba_source": aba_by_L[L][2],
                    }
                    for L in write_layers
                },
                "T": int(emb["T"]),
                "n_changed": int(emb["n_changed"]),
                "per_layer": per_layer,
                "gapA": gap_sum,
                "gapA_pair0": float(per_layer[0]["gapA"]),
                "gapA_pair1": float(per_layer[1]["gapA"]),
                "gapA_sum": gap_sum,
                "wm_cosine_sum": wm_cos_sum,
                "orig_cosine_sum": orig_cos_sum,
                "gapA_write": gap_write_sum,
                "gapA_write_pair0": float(per_layer[0]["gapA_write"]),
                "gapA_write_pair1": float(per_layer[1]["gapA_write"]),
                "gapA_sum_write": gap_write_sum,
                "write_cosine_sum": write_cos_sum,
                "token_agree_wm": float(np.mean([x["token_agree_wm"] for x in per_layer])),
                "token_agree_write": float(np.mean([x["token_agree_write"] for x in per_layer])),
                "wave_corr": _wave_corr(wav, wm),
                "lsd": q["lsd"],
                "mcd": q["mcd"],
                "sr": int(sr),
                "out_sr": int(out_sr),
                "detect": "band_zq_two_layer_cos_sum",
            }
        )
    return rows


def _worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    pin_npu(str(payload["device"]))
    log_fp, log = open_worker_log(payload)
    codec = str(payload["codec"]).lower()
    handle, sr = load_codec(codec, payload)
    pool = pool_for(codec, handle)
    table = None
    table_path = str(payload.get("table_path") or "").strip()
    if table_path:
        try:
            table = ensure_rare_table(path=table_path)
        except Exception as exc:
            log(f"WARN rare table unavailable: {exc}")

    per_clip_dir = Path(payload["per_clip_dir"])
    per_clip_dir.mkdir(parents=True, exist_ok=True)
    qtmp = Path(payload.get("wav_out") or (per_clip_dir / "_qtmp"))
    qtmp.mkdir(parents=True, exist_ok=True)
    try:
        lsd_tool, mcd_tool = _make_quality_tools(int(sr))
    except Exception as exc:
        log(f"WARN quality tools unavailable: {type(exc).__name__}: {exc}")
        lsd_tool, mcd_tool = None, None

    aba_mode = str(payload.get("aba_mode") or "rare")
    a_cli = int(payload.get("a", -1))
    b_cli = int(payload.get("b", -1))
    token_seed = int(payload.get("token_seed") or 42)
    pairs = all_layer_pairs(pool)
    max_pairs = int(payload.get("max_pairs") or 0)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    log(
        f"pool=L{pool[0]}-L{pool[-1]} ({len(pool)} layers) "
        f"n_pairs={len(pairs)} aba={aba_mode}"
    )

    rows: list[dict] = []
    n_err = 0
    for tag, path_str in payload["items"]:
        out_p = per_clip_dir / f"{tag}.json"
        if out_p.exists() and bool(payload.get("resume")):
            try:
                prev = json.loads(out_p.read_text(encoding="utf-8"))
                rows.extend(prev.get("rows") or [])
                log(f"resume {tag}")
                continue
            except Exception:
                pass
        try:
            wav = np.asarray(read_wav(path_str, sr), dtype=np.float32).reshape(-1)
            clip_rows = process_clip_two_layer(
                codec,
                handle,
                wav,
                sr=sr,
                pairs=pairs,
                table=table,
                aba_mode=aba_mode,
                a_cli=a_cli,
                b_cli=b_cli,
                token_seed=token_seed,
                tmp_dir=qtmp,
                clip_stem=tag,
                lsd_tool=lsd_tool,
                mcd_tool=mcd_tool,
            )
            dump_json(
                out_p,
                {
                    "name": tag,
                    "path": path_str,
                    "codec": codec,
                    "layer_pool": pool,
                    "n_pairs": len(pairs),
                    "pairs": pairs,
                    "aba_policy": "per_layer_rare",
                    "rows": clip_rows,
                },
            )
            rows.extend(clip_rows)
            last = clip_rows[-1] if clip_rows else {}
            log(
                f"ok {codec} {tag} n_pairs={len(clip_rows)} "
                f"last={last.get('pair_key')} gapA={_fmt(last.get('gapA'))} "
                f"gapA_write={_fmt(last.get('gapA_write'))} "
                f"gapA_sum={_fmt(last.get('gapA_sum'))} "
                f"agree={_fmt(last.get('token_agree_wm'))} "
                f"corr={_fmt(last.get('wave_corr'))} lsd={_fmt(last.get('lsd'))} "
                f"mcd={_fmt(last.get('mcd'))}"
            )
        except Exception as exc:
            n_err += 1
            log(f"FAIL {codec} {tag}: {type(exc).__name__}: {exc}")

    del handle
    free_device()
    if log_fp is not None:
        log_fp.close()
    return {
        "worker_id": int(payload["worker_id"]),
        "codec": codec,
        "rows": rows,
        "n_ok": len({r.get("name") for r in rows}),
        "n_error": int(n_err),
        "layer_pool": pool,
    }


def aggregate(rows: list[dict]) -> dict:
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault(str(r.get("pair_key") or "?"), []).append(r)

    def _blk(xs: list[dict]) -> dict:
        return {
            "n": len(xs),
            "mean_gapA": _mean([r["gapA"] for r in xs]),
            "mean_gapA_sum": _mean([r.get("gapA_sum", r["gapA"]) for r in xs]),
            "mean_gapA_write": _mean([r.get("gapA_write") for r in xs]),
            "mean_gapA_pair0": _mean([r.get("gapA_pair0") for r in xs]),
            "mean_gapA_pair1": _mean([r.get("gapA_pair1") for r in xs]),
            "mean_gapA_write_pair0": _mean([r.get("gapA_write_pair0") for r in xs]),
            "mean_gapA_write_pair1": _mean([r.get("gapA_write_pair1") for r in xs]),
            "mean_token_agree_wm": _mean([r["token_agree_wm"] for r in xs]),
            "mean_token_agree_write": _mean([r.get("token_agree_write") for r in xs]),
            "mean_wave_corr": _mean([r["wave_corr"] for r in xs]),
            "mean_lsd": _mean([r["lsd"] for r in xs]),
            "mean_mcd": _mean([r["mcd"] for r in xs]),
        }

    return {
        "overall": _blk(rows),
        "by_pair": {k: _blk(v) for k, v in sorted(by_pair.items())},
        "n_rows": len(rows),
    }


def write_codec_findings(path: Path, codec: str, agg: dict, pool: list[int]) -> None:
    o = agg.get("overall") or {}
    lines = [
        f"# {codec} · 二层随机 band ABA\n\n",
        f"- 层池：L{pool[0]}–L{pool[-1]}\n",
        f"- 总体：gapA_sum={_fmt(o.get('mean_gapA_sum'))} "
        f"(=Σcos_wm−Σcos_orig) gapA_write={_fmt(o.get('mean_gapA_write'))} "
        f"wave_corr={_fmt(o.get('mean_wave_corr'))} LSD={_fmt(o.get('mean_lsd'))} "
        f"MCD={_fmt(o.get('mean_mcd'))} n={o.get('n')}\n",
        "- 每层各自算 cos，总 gap = Σ cos(wm) − Σ cos(orig)\n\n",
        "| pair | n | gapA_sum | gap0 | gap1 | gapA_write | agree_wm | wave corr | LSD | MCD |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for k, m in (agg.get("by_pair") or {}).items():
        lines.append(
            f"| {k} | {m['n']} | {_fmt(m.get('mean_gapA_sum'))} | "
            f"{_fmt(m.get('mean_gapA_pair0'))} | {_fmt(m.get('mean_gapA_pair1'))} | "
            f"{_fmt(m.get('mean_gapA_write'))} | {_fmt(m.get('mean_token_agree_wm'))} | "
            f"{_fmt(m['mean_wave_corr'])} | {_fmt(m['mean_lsd'])} | {_fmt(m['mean_mcd'])} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def write_combined_findings(path: Path, systems: dict) -> None:
    lines = [
        "# codecs band two-layer random\n\n",
        "gapA_sum = Σ cos(wm,tmpl_L) − Σ cos(orig,tmpl_L)\n\n",
        "| codec | pool | gapA_sum | gapA_write | wave corr | LSD | MCD | n |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for codec, blk in systems.items():
        o = (blk.get("aggregate") or {}).get("overall") or {}
        pool = blk.get("layer_pool") or LAYER_POOL.get(codec, [])
        pl = f"L{pool[0]}–L{pool[-1]}" if pool else "—"
        lines.append(
            f"| {codec} | {pl} | {_fmt(o.get('mean_gapA_sum'))} | {_fmt(o.get('mean_gapA_write'))} | "
            f"{_fmt(o.get('mean_wave_corr'))} | {_fmt(o.get('mean_lsd'))} | "
            f"{_fmt(o.get('mean_mcd'))} | {o.get('n', '—')} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _load_rows_from_disk(per_clip_dir: Path, clips: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for tag, _ in clips:
        fp = per_clip_dir / f"{tag}.json"
        if fp.exists():
            prev = json.loads(fp.read_text(encoding="utf-8"))
            rows.extend(prev.get("rows") or [])
    return rows


def run_one_codec(codec: str, clips: list[tuple[str, str]], args, out_root: Path) -> dict:
    codec_dir = out_root / codec
    codec_dir.mkdir(parents=True, exist_ok=True)
    (codec_dir / "logs").mkdir(parents=True, exist_ok=True)
    (codec_dir / "wav").mkdir(parents=True, exist_ok=True)
    per_clip_dir = codec_dir / "per_clip"
    per_clip_dir.mkdir(parents=True, exist_ok=True)
    pool = LAYER_POOL[codec]

    n_pairs = len(all_layer_pairs(pool))
    max_pairs = int(getattr(args, "max_pairs", 0) or 0)
    if max_pairs > 0:
        n_pairs = min(n_pairs, max_pairs)
    print(
        f"=== {codec} two-layer-all-pairs n={len(clips)} pool=L{pool[0]}-{pool[-1]} "
        f"n_pairs={n_pairs} workers={args.num_workers} ===",
        flush=True,
    )
    outs = launch_clip_workers(
        _worker,
        clips,
        args,
        out_dir=codec_dir,
        common={
            "codec": codec,
            "per_clip_dir": str(per_clip_dir),
            "resume": bool(args.resume),
            "tokenizer_path": str(args.tokenizer_path),
            "dac_path": str(getattr(args, "dac_path", "") or ""),
            "model_path": str(args.model_path),
            "table_path": str(args.table),
            "aba_mode": str(args.aba_mode),
            "a": int(args.a),
            "b": int(args.b),
            "token_seed": int(args.token_seed),
            "max_pairs": int(getattr(args, "max_pairs", 0) or 0),
        },
    )
    rows = flatten_key(outs, "rows")
    if args.resume:
        disk = _load_rows_from_disk(per_clip_dir, clips)
        if len(disk) >= len(rows):
            rows = disk
    used_pool = pool
    for o in outs:
        if o.get("layer_pool"):
            used_pool = list(o["layer_pool"])
            break
    agg = aggregate(rows)
    dump_json(
        codec_dir / "summary.json",
        {
            "codec": codec,
            "n_clips": len(clips),
            "n_rows": len(rows),
            "layer_pool": used_pool,
            "n_pairs": n_pairs,
            "pair_mode": "all_combinations",
            "aba_mode": str(args.aba_mode),
            "aba_policy": "per_layer_rare_same_as_single",
            "detect": "band_zq_two_layer_cos_sum",
            "metrics": [
                "gapA_sum",
                "wm_cosine_sum",
                "orig_cosine_sum",
                "gapA_pair0",
                "gapA_pair1",
                "gapA_write",
                "wave_corr",
                "lsd",
                "mcd",
            ],
            "aggregate": agg,
        },
    )
    write_codec_findings(codec_dir / "FINDINGS.md", codec, agg, used_pool)
    o = agg.get("overall") or {}
    print(
        f"  [{codec}] gapA={_fmt(o.get('mean_gapA'))} gapA_sum={_fmt(o.get('mean_gapA_sum'))} "
        f"corr={_fmt(o.get('mean_wave_corr'))} lsd={_fmt(o.get('mean_lsd'))} "
        f"mcd={_fmt(o.get('mean_mcd'))}",
        flush=True,
    )
    return {"aggregate": agg, "layer_pool": used_pool}


def main() -> None:
    p = make_parser(__doc__, default_out=str(DEFAULT_OUT), with_names=False)
    p.set_defaults(model_path=default_moss_path(), table=default_table_path())
    p.add_argument("--codecs", type=str, default=",".join(DEFAULT_CODECS))
    p.add_argument("--tokenizer-path", type=str, default=default_qwen_tokenizer())
    p.add_argument(
        "--dac-path",
        type=str,
        default="/data1/home/ynian/pred_models/dac/weights_24khz_8kbps_0.0.4.pth",
    )
    p.add_argument("--aba-mode", type=str, default="rare", choices=("rare", "random"))
    p.add_argument("--a", type=int, default=-1)
    p.add_argument("--b", type=int, default=-1)
    p.add_argument("--token-seed", type=int, default=42)
    p.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="调试：每句最多跑前 K 个层对；0=全部组合",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    set_seed(int(args.token_seed))
    if not str(getattr(args, "wav_dir", "") or "").strip() or not Path(args.wav_dir).exists():
        args.wav_dir = default_mix_1500()
    if not Path(str(args.model_path)).exists():
        args.model_path = default_moss_path()
    if not Path(str(args.tokenizer_path)).exists():
        args.tokenizer_path = default_qwen_tokenizer()
    dac_path = str(args.dac_path or "").strip()
    if dac_path and not Path(dac_path).exists():
        raise SystemExit(f"--dac-path not found: {dac_path}")

    codecs = [c.strip().lower() for c in str(args.codecs).split(",") if c.strip()]
    for c in codecs:
        if c not in DEFAULT_CODECS:
            raise SystemExit(f"unknown codec {c!r}")

    wav_dir = Path(args.wav_dir)
    clips = list_mix_clips(wav_dir, limit=int(args.limit))
    if not clips:
        raise SystemExit(f"no audio in {wav_dir}")

    out_dir = resolve_out_dir(args, OUT_STEM)
    print(
        f"codecs={codecs} n={len(clips)} pair_mode=all "
        f"pools=moss0-14/qwen0-8/dac0-10 "
        f"(C=105/36/55) aba={args.aba_mode} workers={args.num_workers}",
        flush=True,
    )

    systems = {}
    for codec in codecs:
        systems[codec] = run_one_codec(codec, clips, args, out_dir)

    dump_json(
        out_dir / "summary.json",
        {
            "experiment": OUT_STEM,
            "wav_dir": str(wav_dir),
            "n_clips": len(clips),
            "pair_mode": "all_combinations",
            "n_pairs_by_codec": {
                c: len(all_layer_pairs(LAYER_POOL[c])) for c in codecs
            },
            "layer_pools": LAYER_POOL,
            "aba_mode": str(args.aba_mode),
            "aba_policy": "per_layer_rare_same_as_single",
            "detect": "band_zq_two_layer",
            "codecs": codecs,
            "systems": systems,
            "metrics": ["gapA", "gapA_sum", "wave_corr", "lsd", "mcd"],
        },
    )
    write_combined_findings(out_dir / "FINDINGS.md", systems)
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    ensure_spawn()
    main()
