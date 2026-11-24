"""MOSS / Qwen3 / DAC：单层 band ABA 替换的逐层 watermark gap 实验。

协议
----
1. encode → **只替换写入层** 为该层 rare ABA 循环（其余层冻结）→ decode
2. **每层用 rare 表对应层的 ABA**（``--a/--b`` 可强制全层同一对）
3. 检测（band）：再编码取该层 ``z_q``，与 ABA 模板 cosine
4. ``gapA = cosine(wm, tmpl) − cosine(orig, tmpl)``
5. 质量：波形 corr、LSD、MCD（相对干净原波形）

例::

  $env:PYTHONPATH = "e:\\code\\water_marks"
  python -m P1.experiments.run_codecs_band_layer_gap \\
    --codecs moss,qwen,dac --layers 0-9 \\
    --wav-dir D:/data/codec_an/data/mix_1500 --limit 3 \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from base.codec_dac import (
    decode_codes as dac_decode,
    encode_codes as dac_encode,
    layer_zq_from_codes as dac_layer_zq,
    load_dac,
)
from base.codec_qwen import (
    TOKENIZER_PATH as QWEN_TOKENIZER_DEFAULT,
    decode_codes_kt,
    encode_codes_kt,
    layer_zq_from_ids as qwen_layer_zq,
    load_qwen_tokenizer,
)
from base.config import (
    EXP_ROOT,
    default_mix_1500,
    default_model_path,
    default_table_path,
    first_existing,
)
from base.exp_parallel import flatten_key, launch_clip_workers, make_parser, resolve_out_dir
from base.io import match_len, write_wav_float
from base.lsd import CalcuLSD
from base.mcd import CalcuMCD
from base.moss_io import decode_moss, encode_moss, read_wav, set_seed
from base.npu_parallel import ensure_spawn, free_device, limit_cpu_threads, open_worker_log, pin_npu
from base.rare_tokens import ensure_rare_table, layer_aba
from base.residual_feat import codes_to_zq_sum, feat_cosine
from base.util import AUDIO_EXTS, dump_json, load_model
from base.wave import cosine_sim

OUT_STEM = "codecs_band_layer_gap"
_LOCAL_EXP = Path(r"/data1/home/ynian/codes/eval/add_exp/result")
DEFAULT_OUT = (EXP_ROOT if EXP_ROOT.exists() else _LOCAL_EXP) / OUT_STEM
DEFAULT_CODECS = ("moss", "qwen", "dac")
PERIOD = 3

# codebook sizes used for random ABA
CB_MOSS = 1024
CB_DAC = 1024
CB_QWEN = 2048

_QWEN_PATHS = (
    Path(QWEN_TOKENIZER_DEFAULT),
    Path(r"/data1/home/ynian/pred_models/Qwen3-TTS-Tokenizer-12Hz"),
    Path(r"/data1/home/ynian/pred_models/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer"),
    Path(r"E:\pred_model\Qwen3-TTS-12Hz-1.7B-Base\speech_tokenizer"),
)


def default_moss_path() -> str:
    return default_model_path()


def default_qwen_tokenizer() -> str:
    return first_existing(*_QWEN_PATHS)


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.4f}"


def _mean(xs: list) -> float | None:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return None if not vals else float(np.mean(vals))


def parse_layers(s: str) -> list[int] | None:
    """``None`` = 用 codec 全部层；``0-9`` / ``0,1,2`` 为显式列表。"""
    s = str(s or "").strip()
    if not s or s.lower() in {"all", "*"}:
        return None
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if str(x).strip() != ""]


def list_mix_clips(wav_dir: Path, limit: int = 0) -> list[tuple[str, str]]:
    files = sorted(p for p in Path(wav_dir).iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if limit > 0:
        files = files[: int(limit)]
    return [(p.stem, str(p)) for p in files]


def make_random_aba(rng: np.random.Generator, codebook: int) -> tuple[int, int]:
    codebook = max(2, int(codebook))
    a = int(rng.integers(0, codebook))
    b = int(rng.integers(0, codebook - 1))
    if b >= a:
        b += 1
    return a, b


def aba_ids_np(T: int, a: int, b: int, period: int = PERIOD) -> np.ndarray:
    motif = np.asarray([int(a), int(b), int(a)], dtype=np.int64)
    if int(period) != 3:
        raise ValueError("only period=3 ABA supported")
    return motif[np.arange(int(T)) % 3]


def aba_ids_torch(T: int, a: int, b: int, device=None) -> torch.Tensor:
    motif = torch.tensor([int(a), int(b), int(a)], device=device, dtype=torch.long)
    idx = torch.arange(int(T), device=device) % 3
    return motif[idx]


def resolve_aba_for_layer(
    *,
    codec: str,
    layer: int,
    table: dict | None,
    aba_mode: str,
    a_cli: int,
    b_cli: int,
    token_seed: int,
) -> tuple[int, int, str]:
    """按写入层取 ABA：默认用该层 rare token；``--a/--b`` 则全层强制同一对。"""
    if int(a_cli) >= 0 and int(b_cli) >= 0 and int(a_cli) != int(b_cli):
        return int(a_cli), int(b_cli), "cli"
    mode = str(aba_mode or "rare").strip().lower()
    cb = {"moss": CB_MOSS, "dac": CB_DAC, "qwen": CB_QWEN}[codec]
    L = int(layer)
    if mode == "rare" and table is not None:
        picked = layer_aba(table, L)
        if picked:
            toks, _ = picked
            a, b = int(toks[0]), int(toks[1])
            if 0 <= a < cb and 0 <= b < cb and a != b:
                return a, b, f"rare_L{L}"
        # 该层缺失时退回 L9
        picked9 = layer_aba(table, 9)
        if picked9:
            toks, _ = picked9
            a, b = int(toks[0]), int(toks[1])
            if 0 <= a < cb and 0 <= b < cb and a != b:
                return a, b, f"rare_L9_fallback_for_L{L}"
    rng = np.random.default_rng(int(token_seed) + {"moss": 1, "qwen": 2, "dac": 3}[codec] * 1000 + L)
    a, b = make_random_aba(rng, cb)
    return a, b, f"random_cb{cb}_L{L}"


def _wave_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(match_len(a, b), dtype=np.float64).reshape(-1)
    return float(cosine_sim(x, y))


def _make_quality_tools(sr: int):
    import warnings

    warnings.filterwarnings("ignore", message=".*长度差.*")
    return CalcuLSD(sr=int(sr)), CalcuMCD()


def wave_mcd_lsd(
    ref: np.ndarray,
    deg: np.ndarray,
    sr: int,
    *,
    tmp_dir: Path,
    stem: str,
    lsd_tool=None,
    mcd_tool=None,
) -> dict:
    import warnings

    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    deg = np.asarray(match_len(ref, deg), dtype=np.float32).reshape(-1)
    out: dict = {"mcd": None, "lsd": None}
    try:
        if lsd_tool is None:
            lsd_tool, _ = _make_quality_tools(sr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val, _ = lsd_tool.compute(ref, deg)
        out["lsd"] = float(val)
    except Exception:
        pass
    try:
        if mcd_tool is None:
            _, mcd_tool = _make_quality_tools(sr)
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        rp = tmp_dir / f"{stem}_ref.wav"
        dp = tmp_dir / f"{stem}_deg.wav"
        write_wav_float(rp, ref, int(sr))
        write_wav_float(dp, deg, int(sr))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["mcd"] = float(mcd_tool.mcd_file(str(rp), str(dp)))
        rp.unlink(missing_ok=True)
        dp.unlink(missing_ok=True)
    except Exception:
        pass
    return out


# ----- load / n_q / codebook -----


def load_codec(codec: str, payload: dict):
    codec = str(codec).lower()
    device = str(payload["device"])
    if codec == "moss":
        model, sr = load_model(payload["model_path"], device)
        return model, int(sr)
    if codec == "qwen":
        tok = load_qwen_tokenizer(payload["tokenizer_path"], device)
        sr = int(tok.feature_extractor.sampling_rate)
        return tok, sr
    if codec == "dac":
        dac_path = str(payload.get("dac_path") or "").strip() or None
        model = load_dac(device, model_path=dac_path)
        return model, int(model.sample_rate)
    raise ValueError(f"unknown codec: {codec}")


def n_q_of(codec: str, handle) -> int:
    codec = str(codec).lower()
    if codec == "moss":
        return int(len(handle.quantizer.quantizers))
    if codec == "qwen":
        return int(handle.model.encoder_valid_num_quantizers)
    return int(handle.quantizer.n_codebooks)


def resolve_layers(codec: str, handle, layers_arg: list[int] | None) -> list[int]:
    n_q = n_q_of(codec, handle)
    if layers_arg is None:
        return list(range(n_q))
    out = [int(L) for L in layers_arg if 0 <= int(L) < n_q]
    if not out:
        raise ValueError(f"{codec}: no valid layers in {layers_arg} (n_q={n_q})")
    return out


# ----- embed / band detect per codec -----


def embed_moss(model, wav: np.ndarray, layer: int, a: int, b: int) -> dict:
    codes = np.asarray(encode_moss(model, wav), dtype=np.int64)
    n_q, T = codes.shape
    layer = int(layer)
    before = codes[layer].copy()
    codes = codes.copy()
    codes[layer] = aba_ids_np(T, a, b)
    n_changed = int(np.sum(codes[layer] != before))
    wm = np.asarray(match_len(wav, decode_moss(model, codes)), dtype=np.float32).reshape(-1)
    return {"wav": wm, "T": int(T), "n_q": int(n_q), "n_changed": n_changed, "out_sr": None}


def embed_qwen(tok, wav: np.ndarray, sr: int, layer: int, a: int, b: int) -> dict:
    codes = encode_codes_kt(tok, wav, sr)
    K, T = codes.shape
    layer = int(layer)
    before = codes[layer].detach().clone()
    codes = codes.clone()
    codes[layer] = aba_ids_torch(T, a, b, device=codes.device)
    n_changed = int((codes[layer] != before).sum().item())
    wm, out_sr = decode_codes_kt(tok, codes)
    wm = np.asarray(match_len(wav, wm), dtype=np.float32).reshape(-1)
    return {"wav": wm, "T": int(T), "n_q": int(K), "n_changed": n_changed, "out_sr": int(out_sr)}


def embed_dac(model, wav: np.ndarray, sr: int, layer: int, a: int, b: int) -> dict:
    codes = dac_encode(model, wav, sr)
    n, T = codes.shape
    layer = int(layer)
    before = codes[layer].clone()
    codes = codes.clone()
    codes[layer] = aba_ids_torch(T, a, b, device=codes.device)
    n_changed = int((codes[layer] != before).sum().item())
    wm = np.asarray(match_len(wav, dac_decode(model, codes)), dtype=np.float32).reshape(-1)
    return {
        "wav": wm,
        "T": int(T),
        "n_q": int(n),
        "n_changed": n_changed,
        "out_sr": int(model.sample_rate),
    }


def band_zq_moss(model, wav: np.ndarray, layer: int) -> np.ndarray:
    codes = np.asarray(encode_moss(model, wav), dtype=np.int64)
    return codes_to_zq_sum(model, codes, int(layer), int(layer), layers=[int(layer)])


def band_tmpl_moss(model, T: int, layer: int, a: int, b: int) -> np.ndarray:
    n_q = int(len(model.quantizer.quantizers))
    grid = np.zeros((n_q, int(T)), dtype=np.int64)
    grid[int(layer)] = aba_ids_np(T, a, b)
    return codes_to_zq_sum(model, grid, int(layer), int(layer), layers=[int(layer)])


def band_zq_qwen(tok, wav: np.ndarray, sr: int, layer: int) -> np.ndarray:
    codes = encode_codes_kt(tok, wav, sr)
    return qwen_layer_zq(tok, codes[int(layer)], int(layer))


def band_tmpl_qwen(tok, T: int, layer: int, a: int, b: int) -> np.ndarray:
    device = next(tok.model.parameters()).device
    ids = aba_ids_torch(T, a, b, device=device)
    return qwen_layer_zq(tok, ids, int(layer))


def band_zq_dac(model, wav: np.ndarray, sr: int, layer: int) -> np.ndarray:
    codes = dac_encode(model, wav, sr)
    return dac_layer_zq(model, codes, int(layer))


def band_tmpl_dac(model, T: int, layer: int, a: int, b: int) -> np.ndarray:
    device = next(model.parameters()).device
    codes = torch.zeros((int(model.quantizer.n_codebooks), int(T)), dtype=torch.long, device=device)
    codes[int(layer)] = aba_ids_torch(T, a, b, device=device)
    return dac_layer_zq(model, codes, int(layer))


def detect_gapA_band(feat_o: np.ndarray, feat_w: np.ndarray, tmpl: np.ndarray) -> dict:
    so = feat_cosine(feat_o, tmpl)
    sw = feat_cosine(feat_w, tmpl)
    return {
        "orig_cosine": float(so["cosine"]),
        "wm_cosine": float(sw["cosine"]),
        "gapA": float(sw["cosine"] - so["cosine"]),
        "T_orig": int(feat_o.shape[-1]),
        "T_wm": int(feat_w.shape[-1]),
    }


def token_agree_layer(codes_wm, layer: int, a: int, b: int) -> float:
    """codes: np [n_q,T] or torch [n_q,T]."""
    if hasattr(codes_wm, "detach"):
        row = codes_wm[int(layer)].detach().cpu().numpy().astype(np.int64)
    else:
        row = np.asarray(codes_wm[int(layer)], dtype=np.int64).reshape(-1)
    written = aba_ids_np(int(row.shape[0]), a, b)
    t = min(row.size, written.size)
    if t <= 0:
        return 0.0
    return float((row[:t] == written[:t]).mean())


def process_clip_layers(
    codec: str,
    handle,
    wav: np.ndarray,
    *,
    sr: int,
    layers: list[int],
    table: dict | None,
    aba_mode: str,
    a_cli: int,
    b_cli: int,
    token_seed: int,
    tmp_dir: Path,
    clip_stem: str,
    lsd_tool=None,
    mcd_tool=None,
) -> list[dict]:
    """写入层 L 使用该层 rare ABA（或 ``--a/--b`` 强制统一）。"""
    codec = str(codec).lower()
    rows = []
    for L in layers:
        a, b, aba_src = resolve_aba_for_layer(
            codec=codec,
            layer=L,
            table=table,
            aba_mode=aba_mode,
            a_cli=a_cli,
            b_cli=b_cli,
            token_seed=token_seed,
        )
        if codec == "moss":
            emb = embed_moss(handle, wav, L, a, b)
            wm = emb["wav"]
            feat_o = band_zq_moss(handle, wav, L)
            feat_w = band_zq_moss(handle, wm, L)
            T = int(max(feat_o.shape[-1], feat_w.shape[-1], emb["T"]))
            tmpl = band_tmpl_moss(handle, T, L, a, b)
            codes_w = encode_moss(handle, wm)
            agree = token_agree_layer(codes_w, L, a, b)
            out_sr = int(sr)
        elif codec == "qwen":
            emb = embed_qwen(handle, wav, sr, L, a, b)
            wm = emb["wav"]
            feat_o = band_zq_qwen(handle, wav, sr, L)
            feat_w = band_zq_qwen(handle, wm, int(emb["out_sr"] or sr), L)
            T = int(max(feat_o.shape[-1], feat_w.shape[-1], emb["T"]))
            tmpl = band_tmpl_qwen(handle, T, L, a, b)
            codes_w = encode_codes_kt(handle, wm, int(emb["out_sr"] or sr))
            agree = token_agree_layer(codes_w, L, a, b)
            out_sr = int(emb["out_sr"] or sr)
        else:
            emb = embed_dac(handle, wav, sr, L, a, b)
            wm = emb["wav"]
            feat_o = band_zq_dac(handle, wav, sr, L)
            feat_w = band_zq_dac(handle, wm, sr, L)
            T = int(max(feat_o.shape[-1], feat_w.shape[-1], emb["T"]))
            tmpl = band_tmpl_dac(handle, T, L, a, b)
            codes_w = dac_encode(handle, wm, sr)
            agree = token_agree_layer(codes_w, L, a, b)
            out_sr = int(sr)

        det = detect_gapA_band(feat_o, feat_w, tmpl)
        q = wave_mcd_lsd(
            wav,
            wm,
            int(sr),
            tmp_dir=tmp_dir,
            stem=f"{clip_stem}_L{L}",
            lsd_tool=lsd_tool,
            mcd_tool=mcd_tool,
        )
        rows.append(
            {
                "name": clip_stem,
                "codec": codec,
                "write_layer": int(L),
                "aba": [int(a), int(b), int(a)],
                "aba_source": aba_src,
                "T": int(emb["T"]),
                "n_changed": int(emb["n_changed"]),
                "orig_cosine": det["orig_cosine"],
                "wm_cosine": det["wm_cosine"],
                "gapA": det["gapA"],
                "token_agree_wm": float(agree),
                "wave_corr": _wave_corr(wav, wm),
                "lsd": q["lsd"],
                "mcd": q["mcd"],
                "sr": int(sr),
                "out_sr": int(out_sr),
                "detect": "band_zq",
            }
        )
    return rows


def _worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    pin_npu(str(payload["device"]))
    log_fp, log = open_worker_log(payload)
    codec = str(payload["codec"]).lower()
    handle, sr = load_codec(codec, payload)
    layers_arg = payload.get("layers")
    layers = resolve_layers(codec, handle, layers_arg)
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
    # 预览各层 ABA（per-layer rare）
    preview = []
    for L in layers[: min(8, len(layers))]:
        aa, bb, src = resolve_aba_for_layer(
            codec=codec,
            layer=L,
            table=table,
            aba_mode=aba_mode,
            a_cli=a_cli,
            b_cli=b_cli,
            token_seed=token_seed,
        )
        preview.append(f"L{L}=[{aa},{bb},{aa}]({src})")
    log(f"per-layer ABA preview: {', '.join(preview)}" + (" ..." if len(layers) > 8 else ""))

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
            clip_rows = process_clip_layers(
                codec,
                handle,
                wav,
                sr=sr,
                layers=layers,
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
                    "aba_policy": "per_layer_rare",
                    "rows": clip_rows,
                },
            )
            rows.extend(clip_rows)
            last = clip_rows[-1] if clip_rows else {}
            log(
                f"ok {codec} {tag} layers={len(clip_rows)} "
                f"last_L{last.get('write_layer')} aba={last.get('aba')} "
                f"gapA={_fmt(last.get('gapA'))} "
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
        "layers": layers,
    }


def aggregate(rows: list[dict]) -> dict:
    by_layer: dict[int, list[dict]] = {}
    for r in rows:
        by_layer.setdefault(int(r["write_layer"]), []).append(r)
    out_layers = []
    for L in sorted(by_layer):
        xs = by_layer[L]
        out_layers.append(
            {
                "write_layer": int(L),
                "n": len(xs),
                "mean_gapA": _mean([r["gapA"] for r in xs]),
                "mean_orig_cosine": _mean([r["orig_cosine"] for r in xs]),
                "mean_wm_cosine": _mean([r["wm_cosine"] for r in xs]),
                "mean_token_agree_wm": _mean([r["token_agree_wm"] for r in xs]),
                "mean_wave_corr": _mean([r["wave_corr"] for r in xs]),
                "mean_lsd": _mean([r["lsd"] for r in xs]),
                "mean_mcd": _mean([r["mcd"] for r in xs]),
            }
        )
    return {"by_layer": out_layers, "n_rows": len(rows)}


def write_codec_findings(path: Path, codec: str, agg: dict) -> None:
    lines = [
        f"# {codec} · band ABA 单层替换 · 逐层 gap\n\n",
        "| 写入层 | gapA | wm cos | agree | wave corr | LSD | MCD |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for m in agg.get("by_layer") or []:
        lines.append(
            f"| L{m['write_layer']} | {_fmt(m['mean_gapA'])} | {_fmt(m['mean_wm_cosine'])} | "
            f"{_fmt(m['mean_token_agree_wm'])} | {_fmt(m['mean_wave_corr'])} | "
            f"{_fmt(m['mean_lsd'])} | {_fmt(m['mean_mcd'])} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def write_combined_findings(path: Path, systems: dict) -> None:
    lines = [
        "# codecs band layer gap（单层 ABA · band_zq 检测）\n\n",
        "指标：gapA / wave corr / LSD / MCD\n\n",
    ]
    # union of layers
    all_L = sorted(
        {
            int(m["write_layer"])
            for agg in systems.values()
            for m in (agg.get("by_layer") or [])
        }
    )
    for metric, key in (
        ("gapA", "mean_gapA"),
        ("wave_corr", "mean_wave_corr"),
        ("LSD", "mean_lsd"),
        ("MCD", "mean_mcd"),
    ):
        lines.append(f"## {metric}\n\n")
        hdr = "| 层 | " + " | ".join(systems.keys()) + " |\n"
        sep = "|---|" + "|".join(["---:"] * len(systems)) + "|\n"
        lines.append(hdr)
        lines.append(sep)
        for L in all_L:
            cells = []
            for codec, agg in systems.items():
                row = next(
                    (x for x in (agg.get("by_layer") or []) if int(x["write_layer"]) == L),
                    None,
                )
                cells.append(_fmt(None if row is None else row.get(key)))
            lines.append(f"| L{L} | " + " | ".join(cells) + " |\n")
        lines.append("\n")
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

    layers_arg = parse_layers(str(getattr(args, "layers", "") or ""))
    print(
        f"=== {codec} band layer-gap n={len(clips)} layers={layers_arg or 'all'} "
        f"workers={args.num_workers} ===",
        flush=True,
    )
    outs = launch_clip_workers(
        _worker,
        clips,
        args,
        out_dir=codec_dir,
        common={
            "codec": codec,
            "layers": layers_arg,
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
        },
    )
    rows = flatten_key(outs, "rows")
    if args.resume:
        disk = _load_rows_from_disk(per_clip_dir, clips)
        if len(disk) >= len(rows):
            rows = disk
    # 汇总每层实际使用的 ABA（取首条该层记录）
    aba_by_layer = {}
    for r in rows:
        L = int(r["write_layer"])
        if L not in aba_by_layer:
            aba_by_layer[L] = {"aba": r.get("aba"), "aba_source": r.get("aba_source")}
    agg = aggregate(rows)
    dump_json(
        codec_dir / "summary.json",
        {
            "codec": codec,
            "n_clips": len(clips),
            "n_rows": len(rows),
            "layers": layers_arg,
            "aba_by_layer": {f"L{k}": v for k, v in sorted(aba_by_layer.items())},
            "aba_mode": str(args.aba_mode),
            "note": "每层使用 rare 表该层 ABA（或 --a/--b 强制统一）",
            "detect": "band_zq",
            "aggregate": agg,
        },
    )
    write_codec_findings(codec_dir / "FINDINGS.md", codec, agg)
    for m in agg.get("by_layer") or []:
        print(
            f"  [{codec}] L{m['write_layer']} gapA={_fmt(m['mean_gapA'])} "
            f"corr={_fmt(m['mean_wave_corr'])} lsd={_fmt(m['mean_lsd'])} "
            f"mcd={_fmt(m['mean_mcd'])} agree={_fmt(m['mean_token_agree_wm'])}",
            flush=True,
        )
    return agg


def main() -> None:
    p = make_parser(__doc__, default_out=str(DEFAULT_OUT), with_names=False)
    p.set_defaults(model_path=default_moss_path(), table=default_table_path())
    p.add_argument(
        "--codecs",
        type=str,
        default=",".join(DEFAULT_CODECS),
        help="comma list: moss,qwen,dac",
    )
    p.add_argument(
        "--layers",
        type=str,
        default="all",
        help="写入层：0-9 / 0,1,2 / all（各 codec 全部层）",
    )
    p.add_argument(
        "--tokenizer-path",
        type=str,
        default=default_qwen_tokenizer(),
        help="Qwen3 speech_tokenizer 本地目录",
    )
    p.add_argument(
        "--dac-path",
        type=str,
        default="/data1/home/ynian/pred_models/dac/weights_24khz_8kbps_0.0.4.pth",
        help="DAC 权重；空则 download 24khz",
    )
    p.add_argument(
        "--aba-mode",
        type=str,
        default="rare",
        choices=("rare", "random"),
        help="ABA：默认每层用 rare 表该层 token；random=每层按 seed 随机",
    )
    p.add_argument("--a", type=int, default=-1, help="显式 ABA token A（与 --b 同时给则全层强制同一对）")
    p.add_argument("--b", type=int, default=-1, help="显式 ABA token B")
    p.add_argument("--token-seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 句；0=全量")
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
            raise SystemExit(f"unknown codec {c!r}; choose from {DEFAULT_CODECS}")

    wav_dir = Path(args.wav_dir)
    clips = list_mix_clips(wav_dir, limit=int(args.limit))
    if not clips:
        raise SystemExit(f"no audio in {wav_dir}")

    out_dir = resolve_out_dir(args, OUT_STEM)
    print(
        f"codecs={codecs} n={len(clips)} layers={args.layers} "
        f"aba={args.aba_mode} workers={args.num_workers} wav_dir={wav_dir}",
        flush=True,
    )

    systems = {}
    for codec in codecs:
        systems[codec] = run_one_codec(codec, clips, args, out_dir)

    summary = {
        "experiment": OUT_STEM,
        "wav_dir": str(wav_dir),
        "n_clips": len(clips),
        "layers": str(args.layers),
        "aba_mode": str(args.aba_mode),
        "aba_policy": "per_layer_rare",
        "a": int(args.a),
        "b": int(args.b),
        "token_seed": int(args.token_seed),
        "detect": "band_zq",
        "codecs": codecs,
        "systems": systems,
        "metrics": ["gapA", "wave_corr", "lsd", "mcd"],
    }
    dump_json(out_dir / "summary.json", summary)
    write_combined_findings(out_dir / "FINDINGS.md", systems)
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    ensure_spawn()
    main()
