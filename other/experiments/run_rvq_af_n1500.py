"""mix_1500 上的实验 A–F。MOSS、Qwen、DAC。8 进程 × 8 NPU。

每个实验一个文件夹，每条语音一个 JSON::

  {out}/{A|B|C|D|E|F}/{moss|qwen|dac}/{stem}.json

已有 JSON 的语音会跳过。E、F 比 A–D 慢：E 要重算后层，F 要解码。

例::

  python -m other.experiments.run_rvq_af_n1500 \\
    --codecs moss,qwen,dac --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from base.codec_dac import load_dac, wav_to_batch
from base.codec_qwen import (
    decode_codes_kt,
    embeddings,
    encode_codes_kt,
    load_qwen_tokenizer,
    waveform_values,
)
from base.config import default_mix_1500, default_model_path, first_existing
from base.moss_io import read_wav
from base.residual_feat import encode_hidden
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
from base.util import dump_json, load_model
from base.wave import cosine_sim
from other.experiments.run_feat_retain_vs_rcl_n1500 import list_mix_clips
from other.experiments.rvq_walk import walk_rvq, wave_lsd

EXPERIMENTS = ("A", "B", "C", "D", "E", "F")
E_LAYERS = (0, 1, 2, 9)
E_ALPHAS = (0.25, 0.5)
F_SCALES = (0.1, 0.25)
DAC_PATH = Path("/data1/home/ynian/pred_models/dac/weights_24khz_8kbps_0.0.4.pth")
WAV_PATH = Path("/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500")
QWEN_PATH = first_existing(
    Path(r"/data1/home/ynian/pred_models/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer"),
    Path(r"E:\pred_model\Qwen3-TTS-12Hz-1.7B-Base\speech_tokenizer"),
)

OUT_STEM = "rvq_af_n1500"
_OUT_CANDIDATES = Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
DEFAULT_OUT = Path(
    next((str(p.parent / OUT_STEM) for p in _OUT_CANDIDATES if p.parent.exists()), None)
    or _OUT_CANDIDATES[-1]
)

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _tolist(a) -> list:
    return np.asarray(a, dtype=np.float64).tolist()


def unit_cols(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=0, keepdims=True)
    return v / np.maximum(n, 1e-12)


def frame_mean(v: np.ndarray) -> float:
    return float(np.mean(v))


def pack_ratio(values: np.ndarray, weights: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return {
        "mean": float(np.mean(v)),
        "ratio_of_totals": float(np.sum(v * w) / np.sum(w)),
        "p10": float(np.percentile(v, 10)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
    }


def experiment_a(q: np.ndarray, z: np.ndarray) -> dict:
    """q [L,D,T], z [D,T]。逐帧能量与正负交叉项。"""
    z2 = np.sum(z * z, axis=0) + 1e-20
    q2 = np.sum(q * q, axis=1)
    e1_num = np.sum(q2, axis=0)
    big = np.sum(q, axis=0)
    qn2 = np.sum(big * big, axis=0)
    resid = z - big
    r2 = np.sum(resid * resid, axis=0)
    qr = np.sum(big * resid, axis=0)
    pos = np.zeros(z.shape[1], dtype=np.float64)
    neg = np.zeros(z.shape[1], dtype=np.float64)
    n_q = int(q.shape[0])
    for i in range(n_q):
        for j in range(i + 1, n_q):
            cij = 2.0 * np.sum(q[i] * q[j], axis=0)
            pos += np.maximum(cij, 0.0)
            neg += np.minimum(cij, 0.0)
    cross_num = qn2 - e1_num
    terms = {
        "E1": pack_ratio(e1_num / z2, z2),
        "E2": pack_ratio(qn2 / z2, z2),
        "E3": pack_ratio(r2 / z2, z2),
        "Ecross": pack_ratio(cross_num / z2, z2),
        "Epos": pack_ratio(pos / z2, z2),
        "Eneg": pack_ratio(neg / z2, z2),
        "Eqr": pack_ratio((2.0 * qr) / z2, z2),
    }
    return {"n_frames": int(z.shape[1]), "n_layers": n_q, **terms}


def experiment_b(q: np.ndarray, rng: np.random.Generator) -> dict:
    n_q, _d, t = q.shape
    g = np.zeros((n_q, n_q), dtype=np.float64)
    c = np.zeros((n_q, n_q), dtype=np.float64)
    neg = np.zeros((n_q, n_q), dtype=np.float64)
    norms = np.linalg.norm(q, axis=1) + 1e-12
    for i in range(n_q):
        for j in range(i, n_q):
            dot = np.sum(q[i] * q[j], axis=0)
            cos = dot / (norms[i] * norms[j])
            g[i, j] = g[j, i] = float(np.mean(dot))
            c[i, j] = c[j, i] = float(np.mean(cos))
            if i != j:
                neg[i, j] = neg[j, i] = float(np.mean(np.minimum(dot, 0.0)))
    by_dist = []
    for dist in range(1, n_q):
        vals = [float(neg[i, i + dist]) for i in range(n_q - dist)]
        by_dist.append({"dist": dist, "mean_neg_inner": float(np.mean(vals))})
    shuffled = np.empty_like(q)
    for i in range(n_q):
        shuffled[i] = q[i][:, rng.permutation(t)]
    return {
        "n_frames": int(t),
        "mean_cosine": _tolist(c),
        "mean_inner": _tolist(g),
        "mean_neg_inner": _tolist(neg),
        "neg_by_distance": by_dist,
        "shuffle_mean_cosine_offdiag": _offdiag_mean(shuffled),
    }


def _offdiag_mean(q: np.ndarray) -> float:
    n_q = q.shape[0]
    norms = np.linalg.norm(q, axis=1) + 1e-12
    acc, n = 0.0, 0
    for i in range(n_q):
        for j in range(i + 1, n_q):
            cos = np.sum(q[i] * q[j], axis=0) / (norms[i] * norms[j])
            acc += float(np.mean(cos))
            n += 1
    return float(acc / max(n, 1))


def experiment_c(q: np.ndarray, z: np.ndarray, rng: np.random.Generator) -> dict:
    orth = _ortho(q, z)
    shuffled = np.empty_like(q)
    for i in range(q.shape[0]):
        shuffled[i] = q[i][:, rng.permutation(q.shape[-1])]
    orth_s = _ortho(shuffled, z)
    return {
        "n_frames": int(z.shape[1]),
        "cperp_mean": orth["cperp_mean"],
        "cperp_p10": orth["cperp_p10"],
        "cperp_p50": orth["cperp_p50"],
        "cperp_p90": orth["cperp_p90"],
        "cperp_shuffle_mean": orth_s["cperp_mean"],
        "parallel_energy": _tolist(orth["par_e"]),
        "perp_energy": _tolist(orth["perp_e"]),
        "cumulative_orth_norm": _tolist(orth["cum_norm_mean"]),
    }


def _ortho(q: np.ndarray, z: np.ndarray) -> dict:
    n_q, _d, t = q.shape
    z2 = np.sum(z * z, axis=0) + 1e-20
    par_e = np.zeros(n_q)
    perp_e = np.zeros(n_q)
    acc = np.zeros_like(z)
    sum_perp_e = np.zeros(t)
    for i in range(n_q):
        coef = np.sum(q[i] * z, axis=0) / z2
        parallel = z * coef
        perp = q[i] - parallel
        par_e[i] = float(np.mean(np.sum(parallel * parallel, axis=0) / z2))
        perp_e[i] = float(np.mean(np.sum(perp * perp, axis=0) / z2))
        acc += perp
        sum_perp_e += np.sum(perp * perp, axis=0)
    num = np.sum(acc * acc, axis=0)
    cperp = 1.0 - num / np.maximum(sum_perp_e, 1e-20)
    cum = []
    acc = np.zeros_like(z)
    for i in range(n_q):
        coef = np.sum(q[i] * z, axis=0) / z2
        acc += q[i] - z * coef
        cum.append(float(np.mean(np.linalg.norm(acc, axis=0))))
    return {
        "par_e": par_e,
        "perp_e": perp_e,
        "cperp_mean": float(np.mean(cperp)),
        "cperp_p10": float(np.percentile(cperp, 10)),
        "cperp_p50": float(np.percentile(cperp, 50)),
        "cperp_p90": float(np.percentile(cperp, 90)),
        "cum_norm_mean": np.asarray(cum),
    }


def experiment_d(q: np.ndarray, z: np.ndarray) -> dict:
    n_q, _d, t = q.shape
    z2 = np.sum(z * z, axis=0) + 1e-20
    acc = np.zeros_like(z)
    acc_perp = np.zeros_like(z)
    orth = np.zeros((n_q, t))
    rows = []
    prev = z2.copy()
    for i in range(n_q):
        coef = np.sum(q[i] * z, axis=0) / z2
        perp = q[i] - z * coef
        acc += q[i]
        acc_perp += perp
        resid = z - acc
        r2 = np.sum(resid * resid, axis=0)
        d_e = prev - r2
        orth[i] = np.linalg.norm(acc_perp, axis=0)
        rows.append(
            {
                "layer": i,
                "q_over_z": float(np.mean(np.linalg.norm(acc, axis=0) / np.sqrt(z2))),
                "orth_over_z": float(np.mean(orth[i] / np.sqrt(z2))),
                "resid_over_z": float(np.mean(np.sqrt(r2 / z2))),
                "cos_q_z": float(
                    np.mean(
                        np.sum(acc * z, axis=0)
                        / (np.linalg.norm(acc, axis=0) * np.sqrt(z2) + 1e-12)
                    )
                ),
                "dE_pos_frac": float(np.mean(d_e > 0)),
            }
        )
        prev = r2
    peak = orth[0].copy()
    rose = np.zeros(t, dtype=bool)
    rebound = np.zeros(t, dtype=bool)
    for i in range(1, n_q):
        higher = orth[i] > peak
        rebound |= rose & (orth[i] < peak)
        rose |= higher
        peak = np.maximum(peak, orth[i])
    return {"layers": rows, "rebound_frame_frac": float(np.mean(rebound)), "n_frames": int(t)}


def wave_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(int(a.shape[0]), int(b.shape[0]))
    aa = np.asarray(a[:n], dtype=np.float64)
    bb = np.asarray(b[:n], dtype=np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    return float(cosine_sim(aa, bb))


def moss_layers(model, wav: np.ndarray):
    hidden, lengths = encode_hidden(model, wav)
    _codes, zqs, pres = walk_rvq(model, hidden, lengths)
    q = np.stack([np.asarray(v, dtype=np.float64) for v in zqs], axis=0)
    z = np.asarray(pres[0], dtype=np.float64)
    final = np.asarray(pres[-1], dtype=np.float64) - np.asarray(zqs[-1], dtype=np.float64)
    return q, z, final


def trace_moss(model, wav: np.ndarray):
    q, z, _final = moss_layers(model, wav)
    return [{"name": "rvq", "q": q, "z": z}]


def trace_dac(model, wav: np.ndarray, sr: int):
    x = wav_to_batch(model, wav, sr)
    with torch.inference_mode():
        z = model.encoder(x).float()
        residual = z
        qs = []
        for quantizer in model.quantizer.quantizers:
            out = quantizer(residual)
            z_q = out[0].float()
            if z_q.shape[-1] != residual.shape[-1]:
                z_q = z_q.transpose(1, 2)
            qs.append(z_q[0].detach().cpu().numpy())
            residual = residual - z_q
        z_np = z[0].detach().cpu().numpy()
    q = np.stack(qs, axis=0).astype(np.float64)
    return [{"name": "rvq", "q": q, "z": np.asarray(z_np, dtype=np.float64)}]


def _walk_qwen_rvq(rvq, hidden: torch.Tensor, n_layers: int):
    residual = hidden
    if getattr(rvq, "input_proj", None) is not None:
        residual = rvq.input_proj(residual)
    residual = residual.float()
    z = residual[0].detach().cpu().numpy()
    qs = []
    for i in range(int(n_layers)):
        layer = rvq.layers[i]
        idx = layer.encode(residual)
        z_q = layer.decode(idx).float()
        if z_q.shape[-1] != residual.shape[-1]:
            z_q = z_q.transpose(1, 2)
        qs.append(z_q[0].detach().cpu().numpy())
        residual = residual - z_q
    if not qs:
        return None
    q = np.stack(qs, axis=0).astype(np.float64)
    return {"q": q, "z": np.asarray(z, dtype=np.float64)}


def trace_qwen(tokenizer, wav: np.ndarray, sr: int):
    n_q = int(tokenizer.model.encoder_valid_num_quantizers)
    qmod = tokenizer.model.encoder.quantizer
    n_sem = int(qmod.num_semantic_quantizers)
    values = waveform_values(tokenizer, wav, sr)
    with torch.inference_mode():
        emb = embeddings(tokenizer, values)
        chains = []
        sem = _walk_qwen_rvq(qmod.semantic_residual_vector_quantizer, emb, n_sem)
        if sem is not None:
            sem["name"] = "semantic"
            chains.append(sem)
        ac = _walk_qwen_rvq(
            qmod.acoustic_residual_vector_quantizer, emb, n_q - n_sem
        )
        if ac is not None:
            ac["name"] = "acoustic"
            chains.append(ac)
    return chains


def _chain_block(chain: dict, experiment: str, rng) -> dict:
    q, z = chain["q"], chain["z"]
    body = {"chain": chain["name"], "n_layers": int(q.shape[0]), "n_frames": int(q.shape[-1])}
    if experiment == "A":
        body.update(experiment_a(q, z))
    elif experiment == "B":
        body.update(experiment_b(q, rng))
    elif experiment == "C":
        body.update(experiment_c(q, z, rng))
    elif experiment == "D":
        body.update(experiment_d(q, z))
    return body


def save_geom(out_dir: Path, codec: str, stem: str, chains: list[dict], experiments: set[str], rng):
    written = []
    for experiment in ("A", "B", "C", "D"):
        if experiment not in experiments:
            continue
        path = out_dir / experiment / codec / f"{stem}.json"
        if path.exists():
            continue
        write_json(
            path,
            {
                "stem": stem,
                "codec": codec,
                "experiment": experiment,
                "chains": [_chain_block(chain, experiment, rng) for chain in chains],
            },
        )
        written.append(str(path))
    return written


def moss_decode_sum(model, q_sum: np.ndarray, wav_len: int) -> np.ndarray:
    device = next(model.parameters()).device
    emb = torch.from_numpy(np.ascontiguousarray(q_sum)).float().to(device).unsqueeze(0)
    with torch.inference_mode():
        d = model.quantizer.output_proj(emb)
        lengths = torch.tensor([q_sum.shape[-1]], device=device)
        for module in model.decoder:
            d, lengths = module(d, lengths)
    audio = np.asarray(d.detach().float().cpu().numpy()).reshape(-1)
    return audio[:wav_len]


def dac_decode_sum(model, q_sum: np.ndarray, wav_len: int) -> np.ndarray:
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    z = torch.from_numpy(np.ascontiguousarray(q_sum)).float().to(device).unsqueeze(0)
    with torch.inference_mode():
        y = model.decode(z)
    audio = y[0, 0].detach().float().cpu().numpy().reshape(-1)
    return audio[:wav_len]


def experiment_e_moss(model, wav: np.ndarray, rng: np.random.Generator) -> dict:
    hidden, lengths = encode_hidden(model, wav)
    quantizer = model.quantizer
    z = quantizer.input_proj(hidden)
    batch, _, max_time = z.shape
    t_use = min(int(lengths[0].item()), int(max_time))
    mask = torch.arange(max_time, device=z.device).expand(batch, max_time) < t_use
    residual = z.clone().float()
    qs = []
    with torch.inference_mode():
        for layer in quantizer.quantizers:
            z_q, _idx, _ = layer(residual * mask.unsqueeze(1))
            qs.append(z_q)
            residual = residual - z_q * mask.unsqueeze(1)
    base = _sum_torch(qs, t_use)
    base_wav = moss_decode_sum(model, base, wav.shape[0])
    rows = []
    sr = int(getattr(model, "sampling_rate", None) or getattr(model.config, "sample_rate", 24000))
    for layer in E_LAYERS:
        if layer >= len(qs):
            continue
        rows.extend(_perturb_layers(model, z, mask, qs, layer, t_use, base, base_wav, sr, wav.shape[0], rng, "moss"))
    return {"rows": rows}


def _sum_torch(qs, t_use: int) -> np.ndarray:
    acc = None
    for q in qs:
        part = q[0, :, :t_use].detach().float().cpu().numpy()
        acc = part if acc is None else acc + part
    return acc


def _perturb_layers(model, z, mask, qs, layer, t_use, base, base_wav, sr, wav_len, rng, kind: str):
    q_np = qs[layer][0, :, :t_use].detach().float().cpu().numpy()
    qn = np.linalg.norm(q_np, axis=0, keepdims=True) + 1e-12
    along = q_np / qn
    gauss = rng.normal(size=along.shape)
    gauss = gauss - along * np.sum(gauss * along, axis=0, keepdims=True)
    ortho = unit_cols(gauss)
    enter = z[0, :, :t_use].detach().float().cpu().numpy()
    for prev in qs[:layer]:
        enter = enter - prev[0, :, :t_use].detach().float().cpu().numpy()
    dirs = {"along_q": along, "ortho": ortho, "along_residual": unit_cols(enter)}
    rows = []
    quantizers = model.quantizer.quantizers
    for dname, direction in dirs.items():
        for alpha in E_ALPHAS:
            delta = float(alpha) * qn * direction
            new_q = qs[layer].clone()
            new_q[0, :, :t_use] = torch.from_numpy(np.ascontiguousarray(q_np + delta)).to(new_q.device)
            fixed = [v.clone() for v in qs]
            fixed[layer] = new_q
            recomputed = _requantize_torch(quantizers, z, mask, qs, layer, new_q)
            for mode, chosen in (("fixed", fixed), ("requant", recomputed)):
                total = _sum_torch(chosen, t_use)
                if kind == "moss":
                    wav = moss_decode_sum(model, total, wav_len)
                else:
                    wav = dac_decode_sum(model, total, wav_len)
                diff = total - base
                rows.append(
                    {
                        "layer": layer,
                        "direction": dname,
                        "alpha": float(alpha),
                        "mode": mode,
                        "repr_ratio": float(np.linalg.norm(diff) / (np.linalg.norm(base) + 1e-12)),
                        "cos_q": float(np.sum(total * base) / ((np.linalg.norm(total) * np.linalg.norm(base)) + 1e-12)),
                        "wave_corr": wave_corr(base_wav, wav),
                        "lsd": float(wave_lsd(base_wav, wav, sr)),
                    }
                )
    return rows


def _requantize_torch(quantizers, z, mask, qs, layer, new_q):
    residual = z.clone().float()
    out = []
    with torch.inference_mode():
        for i, quantizer in enumerate(quantizers):
            if i < layer:
                z_q = qs[i]
            elif i == layer:
                z_q = new_q
            else:
                z_q = quantizer(residual * mask.unsqueeze(1))[0].float()
            out.append(z_q)
            residual = residual - z_q * mask.unsqueeze(1)
    return out


def experiment_f_from_q(decode_fn, q: np.ndarray, final_r: np.ndarray, wav_len: int, sr: int, shared: np.ndarray | None):
    rms = float(np.sqrt(np.mean(np.sum(q * q, axis=0))))
    base = decode_fn(q, wav_len)
    own = unit_cols(final_r.mean(axis=1).reshape(-1, 1)).reshape(-1)
    qdir = unit_cols(q.mean(axis=1).reshape(-1, 1)).reshape(-1)
    rng = np.random.default_rng(27)
    ref = shared if shared is not None else own
    gauss = rng.normal(size=ref.shape)
    gauss = gauss - ref * np.dot(gauss, ref)
    rand = gauss / (np.linalg.norm(gauss) + 1e-12)
    dirs = {"own": own, "quant_mean": qdir, "random_orth": rand}
    if shared is not None:
        dirs["shared"] = shared
    rows = []
    for name, direction in dirs.items():
        for scale in F_SCALES:
            for sign in (1.0, -1.0):
                pert = q + (sign * float(scale) * rms) * direction.reshape(-1, 1)
                wav = decode_fn(pert, wav_len)
                rows.append(
                    {
                        "direction": name,
                        "scale": float(scale),
                        "sign": int(sign),
                        "wave_corr": wave_corr(base, wav),
                        "lsd": float(wave_lsd(base, wav, sr)),
                    }
                )
    return {"rows": rows, "mean_residual": own.tolist()}


def worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads", 2)))
    device = str(payload["device"])
    pin_npu(device)
    log_fp, log = open_worker_log(payload)
    experiments = set(payload["experiments"])
    out_dir = Path(payload["out_dir"])
    n_ok, n_err = 0, 0
    try:
        for codec in payload["codecs"]:
            log(f"load {codec} on {device}")
            bundle = _load_codec(codec, device, payload)
            for stem, path in payload["items"]:
                rng = np.random.default_rng(27 + (hash(stem) % 100000))
                try:
                    wav = read_wav(path, bundle["sr"])
                    chains = bundle["trace"](wav)
                    save_geom(out_dir, codec, stem, chains, experiments, rng)
                    if "E" in experiments or "F" in experiments:
                        _save_ef(out_dir, codec, stem, wav, bundle, experiments, rng)
                    n_ok += 1
                    if n_ok % 20 == 0:
                        log(f"{codec} ok={n_ok}")
                except Exception as exc:
                    n_err += 1
                    write_json(
                        out_dir / "errors" / codec / f"{stem}.json",
                        {"stem": stem, "codec": codec, "error": str(exc)},
                    )
                    log(f"ERR {codec} {stem}: {exc}")
            _drop_codec(bundle)
            free_device()
    finally:
        if log_fp is not None:
            log_fp.close()
    return {"worker_id": payload["worker_id"], "n_ok": n_ok, "n_error": n_err}


def _load_codec(codec: str, device: str, payload: dict) -> dict:
    if codec == "moss":
        model, sr = load_model(payload["moss_path"], device)
        return {"kind": "moss", "model": model, "sr": sr, "trace": lambda wav: trace_moss(model, wav)}
    if codec == "dac":
        model = load_dac(device, payload.get("dac_path") or None)
        sr = int(getattr(model, "sample_rate", 24000))
        return {"kind": "dac", "model": model, "sr": sr, "trace": lambda wav: trace_dac(model, wav, sr)}
    if codec == "qwen":
        tok = load_qwen_tokenizer(payload["qwen_path"], device)
        sr = int(tok.feature_extractor.sampling_rate)
        return {"kind": "qwen", "model": tok, "sr": sr, "trace": lambda wav: trace_qwen(tok, wav, sr)}
    raise SystemExit(f"unknown codec {codec}")


def _drop_codec(bundle: dict) -> None:
    bundle.pop("model", None)
    bundle.pop("trace", None)


def _save_ef(out_dir, codec, stem, wav, bundle, experiments, rng):
    kind = bundle["kind"]
    if kind == "qwen":
        if "E" in experiments:
            path = out_dir / "E" / codec / f"{stem}.json"
            if not path.exists():
                write_json(path, {"stem": stem, "codec": codec, "experiment": "E", "rows": _e_qwen(bundle["model"], wav, rng)})
        if "F" in experiments:
            path = out_dir / "F" / codec / f"{stem}.json"
            if not path.exists():
                write_json(
                    path,
                    {
                        "stem": stem,
                        "codec": codec,
                        "experiment": "F",
                        "note": "Qwen 解码入口是离散码。F 需要连续 Q 上加方向后再解码，本文件只保留 A–E 能在码本空间完成的部分。",
                    },
                )
        return
    model = bundle["model"]
    if kind == "moss":
        if "E" in experiments:
            path = out_dir / "E" / codec / f"{stem}.json"
            if not path.exists():
                write_json(path, {"stem": stem, "codec": codec, "experiment": "E", **experiment_e_moss(model, wav, rng)})
        if "F" in experiments:
            path = out_dir / "F" / codec / f"{stem}.json"
            if not path.exists():
                q, _z, final = moss_layers(model, wav)
                q_sum = np.sum(q, axis=0)
                body = experiment_f_from_q(
                    lambda arr, n: moss_decode_sum(model, arr, n),
                    q_sum,
                    final,
                    wav.shape[0],
                    bundle["sr"],
                    None,
                )
                write_json(path, {"stem": stem, "codec": codec, "experiment": "F", **body})
        return
    if "E" in experiments:
        path = out_dir / "E" / codec / f"{stem}.json"
        if not path.exists():
            write_json(path, {"stem": stem, "codec": codec, "experiment": "E", "rows": _e_dac(model, wav, bundle["sr"], rng)})
    if "F" in experiments:
        path = out_dir / "F" / codec / f"{stem}.json"
        if not path.exists():
            chains = trace_dac(model, wav, bundle["sr"])
            q = np.sum(chains[0]["q"], axis=0)
            final = chains[0]["z"] - q
            body = experiment_f_from_q(
                lambda arr, n: dac_decode_sum(model, arr, n),
                q,
                final,
                wav.shape[0],
                bundle["sr"],
                None,
            )
            write_json(path, {"stem": stem, "codec": codec, "experiment": "F", **body})


def _e_dac(model, wav, sr, rng) -> list[dict]:
    x = wav_to_batch(model, wav, sr)
    with torch.inference_mode():
        z = model.encoder(x).float()
    batch, _, max_time = z.shape
    t_use = int(max_time)
    mask = torch.ones(batch, max_time, device=z.device, dtype=torch.bool)
    residual = z.clone()
    qs = []
    with torch.inference_mode():
        for quantizer in model.quantizer.quantizers:
            z_q = quantizer(residual)[0].float()
            if z_q.shape[-1] != residual.shape[-1]:
                z_q = z_q.transpose(1, 2)
            qs.append(z_q)
            residual = residual - z_q
    base = _sum_torch(qs, t_use)
    base_wav = dac_decode_sum(model, base, wav.shape[0])
    rows = []
    for layer in E_LAYERS:
        if layer >= len(qs):
            continue
        rows.extend(
            _perturb_layers(model, z, mask, qs, layer, t_use, base, base_wav, sr, wav.shape[0], rng, "dac")
        )
    return rows


def _e_qwen(tokenizer, wav, rng) -> list[dict]:
    codes = encode_codes_kt(tokenizer, wav, int(tokenizer.feature_extractor.sampling_rate))
    n_q = int(codes.shape[0])
    rows = []
    base_wav, out_sr = decode_codes_kt(tokenizer, codes)
    for layer in E_LAYERS:
        if layer >= n_q:
            continue
        rvq, loc = _qwen_local(tokenizer, layer)
        ids = codes[layer]
        with torch.inference_mode():
            z_q = rvq.layers[loc].decode(ids.view(1, -1)).float()
        q_np = z_q[0].detach().cpu().numpy()
        if q_np.shape[0] < q_np.shape[1]:
            pass
        qn = np.linalg.norm(q_np, axis=0, keepdims=True) + 1e-12
        along = q_np / qn
        for alpha in E_ALPHAS:
            delta = float(alpha) * qn * along
            new_ids = _nearest_qwen(rvq.layers[loc], z_q + torch.from_numpy(delta).to(z_q.device).unsqueeze(0))
            fixed = codes.clone()
            fixed[layer] = new_ids
            wav_f, _ = decode_codes_kt(tokenizer, fixed)
            rows.append(
                {
                    "layer": layer,
                    "direction": "along_q",
                    "alpha": float(alpha),
                    "mode": "fixed",
                    "token_change": float(np.mean(new_ids.detach().cpu().numpy() != ids.detach().cpu().numpy())),
                    "wave_corr": wave_corr(base_wav, wav_f),
                    "lsd": float(wave_lsd(base_wav, wav_f, out_sr)),
                }
            )
    return rows


def _qwen_local(tokenizer, layer: int):
    from base.codec_qwen import _rvq_local

    return _rvq_local(tokenizer, layer)


def _nearest_qwen(layer, z_q: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        idx = layer.encode(z_q.float())
    return idx.view(-1)


def main() -> None:
    ensure_spawn()
    p = argparse.ArgumentParser(description=__doc__)
    add_npu_args(p)
    p.add_argument("--codecs", type=str, default="moss,qwen,dac")
    p.add_argument("--experiments", type=str, default="A,B,C,D,E,F")
    p.add_argument("--wav-dir", type=str, default=WAV_PATH)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--qwen-path", type=str, default=QWEN_PATH)
    p.add_argument("--dac-path", type=str, default=DAC_PATH)
    args = p.parse_args()
    wav_dir = Path(args.wav_dir) if args.wav_dir else Path(default_mix_1500())
    out_dir = Path(args.out_dir) if args.out_dir else Path(r"/data1/home/ynian/codes/eval/add_exp/result/rvq_af_n1500")
    clips = list_mix_clips(wav_dir, int(args.limit))
    if not clips:
        raise SystemExit(f"no audio in {wav_dir}")
    experiments = [x.strip().upper() for x in args.experiments.split(",") if x.strip()]
    codecs = [x.strip() for x in args.codecs.split(",") if x.strip()]
    print(f"clips={len(clips)} codecs={codecs} experiments={experiments} -> {out_dir}", flush=True)
    payloads = build_payloads(
        clips,
        num_workers=int(args.num_workers),
        npu_ids=parse_npu_ids(args.npu_ids),
        device_single=args.device,
        logs_dir=out_dir / "logs",
        common={
            "codecs": codecs,
            "experiments": experiments,
            "out_dir": str(out_dir),
            "cpu_threads": int(args.cpu_threads),
            "moss_path": args.moss_path,
            "qwen_path": args.qwen_path,
            "dac_path": args.dac_path,
        },
    )
    outs = run_shards(worker, payloads, num_workers=int(args.num_workers))
    dump_json(
        out_dir / "launch_summary.json",
        {"n_clips": len(clips), "codecs": codecs, "experiments": experiments, "workers": outs},
    )
    print("done", outs, flush=True)


if __name__ == "__main__":
    main()
