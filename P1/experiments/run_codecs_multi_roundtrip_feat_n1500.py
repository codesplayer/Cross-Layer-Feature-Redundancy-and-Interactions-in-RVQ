"""MOSS / Qwen3 / DAC：多次 encode→decode→encode 同层 z_q 与 encode 连续特征相似度。

**仅依赖本仓库 water_marks**（及 pip：dac / qwen_tts / pymcd / librosa）。
不依赖 codec_an 或其它代码仓库。

路径
----
- **MOSS**：``--model_path``（本地 HF 目录）
- **DAC**：``--dac-path``（可选；空则 ``dac.utils.download(model_type="24khz")``）
- **Qwen3**：``--tokenizer-path``（speech_tokenizer 目录）

例::

  $env:PYTHONPATH = "e:\\code\\water_marks"
  python -m P1.experiments.run_codecs_multi_roundtrip_feat_n1500 \\
    --model_path E:/pred_model/MOSS/MOSS-Audio-Tokenizer \\
    --dac-path "" \\
    --tokenizer-path E:/pred_model/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer \\
    --wav-dir D:/data/codec_an/data/mix_1500 \\
    --codecs moss,qwen,dac --rounds 5 \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from base.codec_dac import (
    decode_codes as dac_decode,
    encode_codes as dac_encode,
    layer_zq_from_codes as dac_layer_zq,
    load_dac,
    wav_to_batch,
)
from base.codec_qwen import (
    TOKENIZER_PATH as QWEN_TOKENIZER_DEFAULT,
    decode_codes_kt,
    embeddings as qwen_embeddings,
    encode_codes_kt,
    layer_zq_from_ids,
    load_qwen_tokenizer,
    waveform_values as qwen_waveform_values,
)
from base.config import (
    EXP_ROOT,
    default_mix_1500,
    default_model_path,
    first_existing,
)
from base.exp_parallel import flatten_key, launch_clip_workers, make_parser, resolve_out_dir
from base.io import match_len, write_wav_float
from base.lsd import CalcuLSD
from base.mcd import CalcuMCD
from base.moss_io import decode_moss, encode_moss, read_wav
from base.npu_parallel import ensure_spawn, free_device, limit_cpu_threads, open_worker_log, pin_npu
from base.residual_feat import codes_to_zq_sum, encode_hidden, feat_cosine
from base.util import AUDIO_EXTS, dump_json, load_model
from base.wave import cosine_sim

OUT_STEM = "codecs_multi_roundtrip_feat_n1500"
_LOCAL_EXP = Path(r"/data1/home/ynian/codes/eval/add_exp/result")
DEFAULT_OUT = (EXP_ROOT if EXP_ROOT.exists() else _LOCAL_EXP) / OUT_STEM
DEFAULT_ROUNDS = 5
DEFAULT_CODECS = ("moss", "qwen", "dac")

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


def list_mix_clips(wav_dir: Path, limit: int = 0) -> list[tuple[str, str]]:
    files = sorted(p for p in Path(wav_dir).iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if limit > 0:
        files = files[: int(limit)]
    return [(p.stem, str(p)) for p in files]


def _codes_np(codes) -> np.ndarray:
    if hasattr(codes, "detach"):
        return codes.detach().cpu().numpy()
    return np.asarray(codes)


def _cos_feat(a: np.ndarray, b: np.ndarray) -> float:
    return float(feat_cosine(a, b)["cosine"])


def _token_agree(c1: np.ndarray, c2: np.ndarray, layer: int | None = None) -> float:
    t = min(int(c1.shape[-1]), int(c2.shape[-1]))
    if layer is None:
        return float((c1[:, :t] == c2[:, :t]).mean())
    return float((c1[int(layer), :t] == c2[int(layer), :t]).mean())


def _wave_cos(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(match_len(a, b), dtype=np.float64).reshape(-1)
    return float(cosine_sim(x, y))


def _make_quality_tools(sr: int):
    """LSD / MCD（本仓库 base.lsd / base.mcd；MCD 需 pip pymcd）。"""
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
    """返回 {mcd, lsd}；失败则为 None。"""
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


# ----- MOSS -----


def moss_continuous(model, wav: np.ndarray) -> np.ndarray:
    hidden, lengths = encode_hidden(model, wav)
    t = int(lengths[0].item())
    return hidden[0, :, :t].detach().float().cpu().numpy()


def moss_zq_layers(model, codes: np.ndarray, n_use: int) -> list[np.ndarray]:
    return [codes_to_zq_sum(model, codes, L, L, layers=[L]) for L in range(int(n_use))]


def _qwen_api():
    return {
        "load": load_qwen_tokenizer,
        "encode": encode_codes_kt,
        "decode": decode_codes_kt,
        "embeddings": qwen_embeddings,
        "waveform_values": qwen_waveform_values,
        "layer_zq": layer_zq_from_ids,
    }


def _dac_api():
    return {
        "load": load_dac,
        "encode": dac_encode,
        "decode": dac_decode,
        "wav_to_batch": wav_to_batch,
        "layer_zq": dac_layer_zq,
    }


def qwen_continuous(api, tok, wav: np.ndarray, sr: int) -> np.ndarray:
    import torch

    values = api["waveform_values"](tok, wav, sr)
    with torch.inference_mode():
        emb = api["embeddings"](tok, values)
    return emb[0].detach().float().cpu().numpy()


def qwen_zq_layers(api, tok, codes_kt, n_use: int) -> list[np.ndarray]:
    return [api["layer_zq"](tok, codes_kt[L], L) for L in range(int(n_use))]


def dac_continuous(api, model, wav: np.ndarray, sr: int) -> np.ndarray:
    import torch

    x = api["wav_to_batch"](model, wav, sr)
    with torch.inference_mode():
        z = model.encoder(x)
    return z[0].detach().float().cpu().numpy()


def dac_zq_layers(api, model, codes_nt, n_use: int) -> list[np.ndarray]:
    return [api["layer_zq"](model, codes_nt, L) for L in range(int(n_use))]


def load_codec(codec: str, payload: dict):
    """Return (handle, native_sr, meta)."""
    codec = str(codec).lower()
    device = str(payload["device"])
    if codec == "moss":
        model, sr = load_model(payload["model_path"], device)
        return model, int(sr), {"api": None}
    if codec == "qwen":
        api = _qwen_api()
        tok = api["load"](payload["tokenizer_path"], device)
        sr = int(tok.feature_extractor.sampling_rate)
        return tok, sr, {"api": api}
    if codec == "dac":
        api = _dac_api()
        dac_path = str(payload.get("dac_path") or "").strip() or None
        model = api["load"](device, model_path=dac_path)
        return model, int(model.sample_rate), {"api": api}
    raise ValueError(f"unknown codec: {codec}")


def n_q_of(codec: str, handle, n_layers: int | None) -> tuple[int, int]:
    codec = str(codec).lower()
    if codec == "moss":
        n_full = int(len(handle.quantizer.quantizers))
    elif codec == "qwen":
        n_full = int(handle.model.encoder_valid_num_quantizers)
    else:
        n_full = int(handle.quantizer.n_codebooks)
    n_q = n_full if n_layers is None else min(int(n_layers), n_full)
    return n_q, n_full


def encode_pack(codec: str, handle, wav: np.ndarray, sr: int, meta: dict, n_q: int):
    codec = str(codec).lower()
    api = meta.get("api")
    if codec == "moss":
        codes = encode_moss(handle, wav)
        cont = moss_continuous(handle, wav)
        zqs = moss_zq_layers(handle, codes, n_q)
        return _codes_np(codes), cont, zqs, int(codes.shape[1])
    if codec == "qwen":
        codes = api["encode"](handle, wav, sr)
        cont = qwen_continuous(api, handle, wav, sr)
        zqs = qwen_zq_layers(api, handle, codes, n_q)
        return _codes_np(codes), cont, zqs, int(codes.shape[1])
    codes = api["encode"](handle, wav, sr)
    cont = dac_continuous(api, handle, wav, sr)
    zqs = dac_zq_layers(api, handle, codes, n_q)
    return _codes_np(codes), cont, zqs, int(codes.shape[1])


def decode_pack(codec: str, handle, codes_np: np.ndarray, meta: dict, ref_wav: np.ndarray, native_sr: int):
    import torch

    codec = str(codec).lower()
    api = meta.get("api")
    if codec == "moss":
        recon = decode_moss(handle, codes_np)
        return np.asarray(match_len(ref_wav, recon), dtype=np.float32).reshape(-1), int(native_sr)
    if codec == "qwen":
        device = next(handle.model.parameters()).device
        codes_t = torch.from_numpy(np.ascontiguousarray(codes_np)).long().to(device)
        recon, out_sr = api["decode"](handle, codes_t)
        return np.asarray(match_len(ref_wav, recon), dtype=np.float32).reshape(-1), int(out_sr)
    device = next(handle.parameters()).device
    codes_t = torch.from_numpy(np.ascontiguousarray(codes_np)).long().to(device)
    recon = api["decode"](handle, codes_t)
    return np.asarray(match_len(ref_wav, recon), dtype=np.float32).reshape(-1), int(native_sr)


def process_clip(
    codec: str,
    handle,
    wav0: np.ndarray,
    *,
    sr: int,
    meta: dict,
    rounds: int,
    n_layers: int | None,
    tmp_dir: Path | None = None,
    clip_stem: str = "clip",
    lsd_tool=None,
    mcd_tool=None,
) -> dict:
    n_q, n_full = n_q_of(codec, handle, n_layers)
    wav = np.asarray(wav0, dtype=np.float32).reshape(-1)
    ref = wav.copy()
    cur_sr = int(sr)
    qdir = Path(tmp_dir) if tmp_dir else Path(".")

    conts: list[np.ndarray] = []
    codes_list: list[np.ndarray] = []
    zq_list: list[list[np.ndarray]] = []
    wavs: list[np.ndarray] = [ref]
    Ts: list[int] = []

    for r in range(int(rounds) + 1):
        codes, cont, zqs, T = encode_pack(codec, handle, wav, cur_sr, meta, n_q)
        codes_list.append(codes)
        conts.append(cont)
        zq_list.append(zqs)
        Ts.append(int(T))
        if r >= int(rounds):
            break
        wav, cur_sr = decode_pack(codec, handle, codes, meta, ref, sr)
        wavs.append(wav)

    hops = []
    for r in range(1, int(rounds) + 1):
        per_layer = []
        for L in range(n_q):
            per_layer.append(
                {
                    "layer": int(L),
                    "zq_cos_vs0": _cos_feat(zq_list[0][L], zq_list[r][L]),
                    "zq_cos_prev": _cos_feat(zq_list[r - 1][L], zq_list[r][L]),
                    "token_agree_vs0": _token_agree(codes_list[0], codes_list[r], L),
                    "token_agree_prev": _token_agree(codes_list[r - 1], codes_list[r], L),
                }
            )
        q_vs0 = wave_mcd_lsd(
            wavs[0],
            wavs[r],
            int(sr),
            tmp_dir=qdir,
            stem=f"{clip_stem}_r{r}_vs0",
            lsd_tool=lsd_tool,
            mcd_tool=mcd_tool,
        )
        q_prev = wave_mcd_lsd(
            wavs[r - 1],
            wavs[r],
            int(sr),
            tmp_dir=qdir,
            stem=f"{clip_stem}_r{r}_prev",
            lsd_tool=lsd_tool,
            mcd_tool=mcd_tool,
        )
        hops.append(
            {
                "round": int(r),
                "T": int(Ts[r]),
                "T0": int(Ts[0]),
                "continuous_cos_vs0": _cos_feat(conts[0], conts[r]),
                "continuous_cos_prev": _cos_feat(conts[r - 1], conts[r]),
                "zq_sum_cos_vs0": _cos_feat(
                    sum(zq_list[0][L] for L in range(n_q)),
                    sum(zq_list[r][L] for L in range(n_q)),
                ),
                "token_agree_all_vs0": _token_agree(codes_list[0][:n_q], codes_list[r][:n_q]),
                "token_agree_all_prev": _token_agree(codes_list[r - 1][:n_q], codes_list[r][:n_q]),
                "wave_cos_vs0": _wave_cos(wavs[0], wavs[r]),
                "wave_cos_prev": _wave_cos(wavs[r - 1], wavs[r]),
                "mcd_vs0": q_vs0["mcd"],
                "mcd_prev": q_prev["mcd"],
                "lsd_vs0": q_vs0["lsd"],
                "lsd_prev": q_prev["lsd"],
                "per_layer": per_layer,
            }
        )
    return {
        "codec": str(codec),
        "n_q": int(n_q),
        "n_q_full": int(n_full),
        "rounds": int(rounds),
        "T0": int(Ts[0]),
        "sr": int(sr),
        "hops": hops,
    }


def _worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    pin_npu(str(payload["device"]))
    log_fp, log = open_worker_log(payload)
    codec = str(payload["codec"]).lower()
    handle, sr, meta = load_codec(codec, payload)
    rounds = int(payload["rounds"])
    n_layers = payload.get("n_layers")
    n_layers = None if n_layers is None else int(n_layers)
    per_clip_dir = Path(payload["per_clip_dir"])
    per_clip_dir.mkdir(parents=True, exist_ok=True)
    qtmp = Path(payload.get("wav_out") or (per_clip_dir / "_qtmp"))
    qtmp.mkdir(parents=True, exist_ok=True)
    try:
        lsd_tool, mcd_tool = _make_quality_tools(int(sr))
    except Exception as exc:
        log(f"WARN quality tools unavailable: {type(exc).__name__}: {exc}")
        lsd_tool, mcd_tool = None, None
    rows = []
    n_err = 0
    for tag, path_str in payload["items"]:
        out_p = per_clip_dir / f"{tag}.json"
        if out_p.exists() and bool(payload.get("resume")):
            try:
                rows.append(json.loads(out_p.read_text(encoding="utf-8")))
                log(f"resume {tag}")
                continue
            except Exception:
                pass
        try:
            wav = np.asarray(read_wav(path_str, sr), dtype=np.float32).reshape(-1)
            rec = process_clip(
                codec,
                handle,
                wav,
                sr=sr,
                meta=meta,
                rounds=rounds,
                n_layers=n_layers,
                tmp_dir=qtmp,
                clip_stem=tag,
                lsd_tool=lsd_tool,
                mcd_tool=mcd_tool,
            )
            rec["name"] = tag
            rec["path"] = path_str
            dump_json(out_p, rec)
            rows.append(rec)
            h1 = rec["hops"][0] if rec["hops"] else {}
            log(
                f"ok {codec} {tag} cont_vs0@1={_fmt(h1.get('continuous_cos_vs0'))} "
                f"agree@1={_fmt(h1.get('token_agree_all_vs0'))} "
                f"mcd={_fmt(h1.get('mcd_vs0'))} lsd={_fmt(h1.get('lsd_vs0'))}"
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
        "n_ok": len(rows),
        "n_error": int(n_err),
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "by_round": []}
    rounds = int(rows[0]["rounds"])
    n_q = int(rows[0]["n_q"])
    by_round = []
    for r in range(1, rounds + 1):
        cont_vs0, cont_prev = [], []
        zq_sum_vs0, agree_vs0, agree_prev = [], [], []
        wave_vs0, wave_prev = [], []
        mcd_vs0, mcd_prev, lsd_vs0, lsd_prev = [], [], [], []
        layer_bags = [
            {"zq_vs0": [], "zq_prev": [], "tok_vs0": [], "tok_prev": []} for _ in range(n_q)
        ]
        for rec in rows:
            hop = next((h for h in rec["hops"] if int(h["round"]) == r), None)
            if not hop:
                continue
            cont_vs0.append(hop["continuous_cos_vs0"])
            cont_prev.append(hop["continuous_cos_prev"])
            zq_sum_vs0.append(hop["zq_sum_cos_vs0"])
            agree_vs0.append(hop["token_agree_all_vs0"])
            agree_prev.append(hop["token_agree_all_prev"])
            wave_vs0.append(hop["wave_cos_vs0"])
            wave_prev.append(hop["wave_cos_prev"])
            mcd_vs0.append(hop.get("mcd_vs0"))
            mcd_prev.append(hop.get("mcd_prev"))
            lsd_vs0.append(hop.get("lsd_vs0"))
            lsd_prev.append(hop.get("lsd_prev"))
            for L, pl in enumerate(hop["per_layer"]):
                if L >= n_q:
                    break
                layer_bags[L]["zq_vs0"].append(pl["zq_cos_vs0"])
                layer_bags[L]["zq_prev"].append(pl["zq_cos_prev"])
                layer_bags[L]["tok_vs0"].append(pl["token_agree_vs0"])
                layer_bags[L]["tok_prev"].append(pl["token_agree_prev"])
        by_round.append(
            {
                "round": r,
                "n": len(cont_vs0),
                "continuous_cos_vs0": _mean(cont_vs0),
                "continuous_cos_prev": _mean(cont_prev),
                "zq_sum_cos_vs0": _mean(zq_sum_vs0),
                "token_agree_all_vs0": _mean(agree_vs0),
                "token_agree_all_prev": _mean(agree_prev),
                "wave_cos_vs0": _mean(wave_vs0),
                "wave_cos_prev": _mean(wave_prev),
                "mcd_vs0": _mean(mcd_vs0),
                "mcd_prev": _mean(mcd_prev),
                "lsd_vs0": _mean(lsd_vs0),
                "lsd_prev": _mean(lsd_prev),
                "per_layer": [
                    {
                        "layer": L,
                        "zq_cos_vs0": _mean(layer_bags[L]["zq_vs0"]),
                        "zq_cos_prev": _mean(layer_bags[L]["zq_prev"]),
                        "token_agree_vs0": _mean(layer_bags[L]["tok_vs0"]),
                        "token_agree_prev": _mean(layer_bags[L]["tok_prev"]),
                    }
                    for L in range(n_q)
                ],
            }
        )
    # layer × round pivot：同层在不同往返上的余弦
    by_layer = []
    for L in range(n_q):
        vs0_by_r, prev_by_r, tok_vs0_by_r, tok_prev_by_r = {}, {}, {}, {}
        for br in by_round:
            pl = next((x for x in br["per_layer"] if int(x["layer"]) == L), None)
            if not pl:
                continue
            r = int(br["round"])
            vs0_by_r[r] = pl["zq_cos_vs0"]
            prev_by_r[r] = pl["zq_cos_prev"]
            tok_vs0_by_r[r] = pl["token_agree_vs0"]
            tok_prev_by_r[r] = pl["token_agree_prev"]
        by_layer.append(
            {
                "layer": L,
                "zq_cos_vs0_by_round": vs0_by_r,
                "zq_cos_prev_by_round": prev_by_r,
                "token_agree_vs0_by_round": tok_vs0_by_r,
                "token_agree_prev_by_round": tok_prev_by_r,
            }
        )
    return {
        "n": len(rows),
        "n_q": n_q,
        "n_q_full": int(rows[0].get("n_q_full") or n_q),
        "rounds": rounds,
        "by_round": by_round,
        "by_layer": by_layer,
    }


def _layer_round_table(agg: dict, key: str) -> list[str]:
    """Markdown: | layer | r1 | r2 | ... | for by_layer[*][key]."""
    rounds = int(agg.get("rounds") or 0)
    lines = [
        "| layer | " + " | ".join(f"r{r}" for r in range(1, rounds + 1)) + " |\n",
        "|---:|" + "|".join(["---:" for _ in range(rounds)]) + "|\n",
    ]
    for row in agg.get("by_layer") or []:
        cells = [_fmt((row.get(key) or {}).get(r)) for r in range(1, rounds + 1)]
        lines.append(f"| {row['layer']} | " + " | ".join(cells) + " |\n")
    return lines


def write_codec_findings(path: Path, codec: str, agg: dict) -> None:
    lines = [
        f"# {codec} 多次编解码：encode 连续特征 / **同层** z_q 相似度（mix_1500）\n\n",
        f"n={agg.get('n')}，rounds={agg.get('rounds')}，层数={agg.get('n_q')}/{agg.get('n_q_full')}\n\n",
        "每层独立：`cos(z_q[L]_r, z_q[L]_0)` 与 `cos(z_q[L]_r, z_q[L]_{r-1})`。\n\n",
        "## 全局（连续特征 / 全层 z_q 和 / 波形质量）\n\n",
        "| round | cont_vs0 | zq_sum_vs0 | tok_agree_vs0 | wave_vs0 | mcd_vs0↓ | lsd_vs0↓ | mcd_prev↓ | lsd_prev↓ |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for br in agg.get("by_round") or []:
        lines.append(
            f"| {br['round']} | {_fmt(br['continuous_cos_vs0'])} | "
            f"{_fmt(br['zq_sum_cos_vs0'])} | {_fmt(br['token_agree_all_vs0'])} | {_fmt(br['wave_cos_vs0'])} | "
            f"{_fmt(br.get('mcd_vs0'))} | {_fmt(br.get('lsd_vs0'))} | "
            f"{_fmt(br.get('mcd_prev'))} | {_fmt(br.get('lsd_prev'))} |\n"
        )
    lines.append("\n## 同层 z_q 余弦 vs 第 0 次（layer × round）\n\n")
    lines.extend(_layer_round_table(agg, "zq_cos_vs0_by_round"))
    lines.append("\n## 同层 z_q 余弦 vs 上一跳（layer × round）\n\n")
    lines.extend(_layer_round_table(agg, "zq_cos_prev_by_round"))
    lines.append("\n## 同层 token_agree vs 第 0 次（layer × round）\n\n")
    lines.extend(_layer_round_table(agg, "token_agree_vs0_by_round"))
    path.write_text("".join(lines), encoding="utf-8")


def write_combined_findings(path: Path, systems: dict) -> None:
    lines = [
        "# MOSS / Qwen3 / DAC 多次编解码特征相似度（mix_1500）\n\n",
        "干净音频，无水印；每个 codec 各自 encode→decode 循环。\n",
        "同层矩阵见各子目录 `FINDINGS.md`（`zq_cos_vs0` / `zq_cos_prev` 的 layer×round）。\n\n",
        "## 第 1 轮 vs 第 0 次编码\n\n",
        "| codec | n | n_q | cont_vs0 | zq_sum_vs0 | tok_agree_vs0 | wave_vs0 | mcd_vs0↓ | lsd_vs0↓ |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for codec in DEFAULT_CODECS:
        if codec not in systems:
            continue
        agg = systems[codec]
        br = (agg.get("by_round") or [{}])[0]
        lines.append(
            f"| {codec} | {agg.get('n')} | {agg.get('n_q')} | {_fmt(br.get('continuous_cos_vs0'))} | "
            f"{_fmt(br.get('zq_sum_cos_vs0'))} | {_fmt(br.get('token_agree_all_vs0'))} | "
            f"{_fmt(br.get('wave_cos_vs0'))} | {_fmt(br.get('mcd_vs0'))} | {_fmt(br.get('lsd_vs0'))} |\n"
        )
    lines.append("\n## 各轮 continuous_cos_vs0\n\n")
    lines.append("| round | " + " | ".join(DEFAULT_CODECS) + " |\n")
    lines.append("|---:|" + "|".join(["---:" for _ in DEFAULT_CODECS]) + "|\n")
    max_r = max((int(s.get("rounds") or 0) for s in systems.values()), default=0)
    for r in range(1, max_r + 1):
        cells = []
        for codec in DEFAULT_CODECS:
            agg = systems.get(codec) or {}
            hop = next((h for h in (agg.get("by_round") or []) if int(h["round"]) == r), None)
            cells.append(_fmt(None if hop is None else hop.get("continuous_cos_vs0")))
        lines.append(f"| {r} | " + " | ".join(cells) + " |\n")

    # 抽几层对照三 codec（末轮 vs0）
    spotlight = (0, 1, 8, 9, 15, 31)
    lines.append("\n## 末轮同层 z_q_cos_vs0 抽层对照\n\n")
    lines.append("| layer | " + " | ".join(DEFAULT_CODECS) + " |\n")
    lines.append("|---:|" + "|".join(["---:" for _ in DEFAULT_CODECS]) + "|\n")
    for L in spotlight:
        cells = []
        for codec in DEFAULT_CODECS:
            agg = systems.get(codec) or {}
            row = next((x for x in (agg.get("by_layer") or []) if int(x["layer"]) == L), None)
            if not row:
                cells.append("—")
                continue
            r_last = int(agg.get("rounds") or 0)
            cells.append(_fmt((row.get("zq_cos_vs0_by_round") or {}).get(r_last)))
        if any(c != "—" for c in cells):
            lines.append(f"| {L} | " + " | ".join(cells) + " |\n")
    path.write_text("".join(lines), encoding="utf-8")


def _load_rows_from_disk(per_clip_dir: Path, clips: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for tag, _ in clips:
        fp = per_clip_dir / f"{tag}.json"
        if fp.exists():
            rows.append(json.loads(fp.read_text(encoding="utf-8")))
    return rows


def run_one_codec(codec: str, clips: list[tuple[str, str]], args, out_root: Path, n_layers) -> dict:
    codec_dir = out_root / codec
    codec_dir.mkdir(parents=True, exist_ok=True)
    (codec_dir / "logs").mkdir(parents=True, exist_ok=True)
    (codec_dir / "wav").mkdir(parents=True, exist_ok=True)
    per_clip_dir = codec_dir / "per_clip"
    per_clip_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== {codec} multi-roundtrip n={len(clips)} rounds={args.rounds} "
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
            "rounds": int(args.rounds),
            "n_layers": n_layers,
            "per_clip_dir": str(per_clip_dir),
            "resume": bool(args.resume),
            "tokenizer_path": str(args.tokenizer_path),
            "dac_path": str(getattr(args, "dac_path", "") or ""),
            "model_path": str(args.model_path),
        },
    )
    rows = flatten_key(outs, "rows")
    if args.resume:
        disk = _load_rows_from_disk(per_clip_dir, clips)
        if len(disk) >= len(rows):
            rows = disk
    agg = aggregate(rows)
    dump_json(
        codec_dir / "summary.json",
        {
            "codec": codec,
            "n_clips": len(clips),
            "n_ok": len(rows),
            "rounds": int(args.rounds),
            "n_layers": n_layers,
            "aggregate": agg,
        },
    )
    write_codec_findings(codec_dir / "FINDINGS.md", codec, agg)
    for br in agg.get("by_round") or []:
        print(
            f"  [{codec}] round={br['round']} cont_vs0={_fmt(br['continuous_cos_vs0'])} "
            f"zq_sum_vs0={_fmt(br['zq_sum_cos_vs0'])} agree={_fmt(br['token_agree_all_vs0'])} "
            f"mcd={_fmt(br.get('mcd_vs0'))} lsd={_fmt(br.get('lsd_vs0'))}",
            flush=True,
        )
    return agg


def main() -> None:
    p = make_parser(__doc__, default_out=str(DEFAULT_OUT), with_names=False)
    p.set_defaults(model_path=default_moss_path())
    p.add_argument(
        "--codecs",
        type=str,
        default=",".join(DEFAULT_CODECS),
        help="comma list: moss,qwen,dac",
    )
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="decode→encode 往返次数")
    p.add_argument(
        "--n-layers",
        type=int,
        default=-1,
        help="统计前多少层 z_q（-1=各 codec 原生全部层）",
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
        default="/data1/home/ynian/pred_models/dac",
        help="DAC 权重路径；空=dac.utils.download(model_type='24khz') 自动缓存",
    )
    p.add_argument("--limit", type=int, default=0, help="调试：只跑前 N 句；0=全量")
    p.add_argument("--resume", action="store_true", help="跳过已有 per_clip json")
    args = p.parse_args()

    if not str(getattr(args, "wav_dir", "") or "").strip() or not Path(args.wav_dir).exists():
        args.wav_dir = default_mix_1500()
    if not Path(str(args.model_path)).exists():
        args.model_path = default_moss_path()
        if not Path(str(args.model_path)).exists():
            args.model_path = default_model_path()
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
    if int(args.limit) <= 0 and len(clips) < 1000:
        print(f"WARN: expected ~1500 clips, got {len(clips)} in {wav_dir}", flush=True)

    out_dir = resolve_out_dir(args, OUT_STEM)
    n_layers = None if int(args.n_layers) < 0 else int(args.n_layers)

    print(
        f"codecs={codecs} n={len(clips)} rounds={args.rounds} "
        f"layers={'all' if n_layers is None else n_layers} "
        f"workers={args.num_workers} wav_dir={wav_dir}",
        flush=True,
    )

    systems = {}
    for codec in codecs:
        systems[codec] = run_one_codec(codec, clips, args, out_dir, n_layers)

    summary = {
        "experiment": OUT_STEM,
        "wav_dir": str(wav_dir),
        "n_clips": len(clips),
        "rounds": int(args.rounds),
        "n_layers": n_layers,
        "codecs": codecs,
        "systems": systems,
    }
    dump_json(out_dir / "summary.json", summary)
    write_combined_findings(out_dir / "FINDINGS.md", systems)
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    ensure_spawn()
    main()
