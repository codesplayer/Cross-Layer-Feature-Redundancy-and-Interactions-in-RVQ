"""多码本单层随机替换 → Layer Impact / Replaceability / Irreplaceability。

对每一层 k：将该层全部帧码字随机替换，再走两路得到累加特征
Z = sum_i z_q^{(i)}；基准是原始语音连续表示 z = input_proj(encoder(wav))::

  - z      ：原始语音连续特征
  - Z_orig ：原始 token 累加（正常量化）
  - Z_rand ：替换后固定后续 token（后层冻结）
  - Z_comp ：减当前层 z_q 后重量化后续层（正确补偿）

逐样本量化基线 s0 = sim(z, Z_orig)（不是 1）::

  D = s0 - sim(z, Z_rand)                              # Layer Impact
  R = (sim(z, Z_comp) - sim(z, Z_rand)) / D             # Replaceability
  U = (s0 - sim(z, Z_comp)) / D                         # Irreplaceability

恒有 R + U = 1（在 D>eps 时）。先逐样本算比例再聚合。
波形同理：c0 = corr(x, x_recon)，再对 rand/comp 算 D/R/U。
D→0 时 R/U 不稳定，仅报告 D（denom_ok=False）。

例::

  python -m watermark.band.experiments.run_layer_replaceability --device cuda --seed 42 \\
    --wav-dir "D:/data/多码本不同层详细分析/实验/data" \\
    --out_dir "D:/data/多码本不同层详细分析/实验/layer_replaceability_zref"

  # 仅用已有 per_clip 重算指标（无需 GPU）::
  python -m watermark.band.experiments.run_layer_replaceability --recompute-only \\
    --out_dir "D:/data/多码本不同层详细分析/实验/layer_replaceability_zref"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from watermark.n3.experiments.run_wm_Lx_comp_sub_n3 import compensate_post_layers, decode_codes
from watermark.future_exp._common import add_model_args, auto_device, dump_json, load_model
from watermark.future_exp._crop_probe_common import default_model_path
from watermark.moss_io import encode_moss, read_wav, set_seed
from watermark.residual_feat import codes_to_zq_sum, encode_hidden, feat_cosine
from watermark.wave import cosine_sim, residual_stats

DEFAULT_WAV_DIR = Path(r"D:/data/多码本不同层详细分析/实验/data")
DEFAULT_OUT = Path(r"D:/data/多码本不同层详细分析/实验/layer_replaceability_zref")
EPS = 1e-8
# D 过小时 R/U 不稳定；仍记录 D，但不报告 R/U
EPS_D = 1e-4


def _mean(xs: list[float]) -> float | None:
    return None if not xs else float(np.mean(xs))


def _std(xs: list[float]) -> float | None:
    return None if not xs else float(np.std(xs))


def _fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{float(x):.4f}"


def feat_corr(a: np.ndarray, b: np.ndarray) -> float:
    t = min(int(a.shape[-1]), int(b.shape[-1]))
    if t <= 0 or a.ndim != 2 or b.ndim != 2:
        return 0.0
    aa = np.asarray(a[:, :t], dtype=np.float64).reshape(-1)
    bb = np.asarray(b[:, :t], dtype=np.float64).reshape(-1)
    if aa.size < 2 or float(np.std(aa)) < EPS or float(np.std(bb)) < EPS:
        return 0.0
    c = float(np.corrcoef(aa, bb)[0, 1])
    return c if np.isfinite(c) else 0.0


def continuous_z(model, wav: np.ndarray) -> np.ndarray:
    """原始语音连续特征 z = input_proj(encoder_hidden) → [D, T]。"""
    hidden, lengths = encode_hidden(model, wav)
    z = model.quantizer.input_proj(hidden)
    t_use = int(lengths[0].item())
    return z[0, :, :t_use].detach().float().cpu().numpy()


def full_zq_sum(model, codes: np.ndarray) -> np.ndarray:
    n_q = int(codes.shape[0])
    return codes_to_zq_sum(model, codes, 0, n_q - 1, layers=list(range(n_q)))


def random_replace_layer(
    codes: np.ndarray,
    layer: int,
    codebook_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """整层随机替换；尽量避开原码字。"""
    out = np.array(codes, dtype=np.int64, copy=True)
    T = int(out.shape[1])
    L = int(layer)
    old = out[L].copy()
    neu = rng.integers(0, int(codebook_size), size=T, dtype=np.int64)
    conflict = neu == old
    if np.any(conflict):
        neu2 = rng.integers(0, int(codebook_size), size=int(np.sum(conflict)), dtype=np.int64)
        still = neu2 == old[conflict]
        if np.any(still) and int(codebook_size) > 1:
            neu2[still] = (neu2[still] + 1) % int(codebook_size)
        neu[conflict] = neu2
    out[L] = neu
    return out


def dru_from_sims(sim_rand: float, sim_comp: float, s0: float) -> dict:
    """相对正常量化基线 s0 的额外损失 D/R/U。"""
    s0 = float(s0)
    sim_rand = float(sim_rand)
    sim_comp = float(sim_comp)
    d = float(s0 - sim_rand)
    out = {
        "s0": s0,
        "sim_rand": sim_rand,
        "sim_comp": sim_comp,
        "D": d,
        "R": float("nan"),
        "U": float("nan"),
        "R_plus_U": float("nan"),
        "denom_ok": False,
    }
    if d < EPS_D:
        return out
    r = float((sim_comp - sim_rand) / d)
    u = float((s0 - sim_comp) / d)
    out.update({"R": r, "U": u, "R_plus_U": float(r + u), "denom_ok": True})
    return out


def wav_corr(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """与补偿实验一致：波形 Pearson corr。"""
    n = min(len(ref), len(deg))
    if n < 2:
        return 0.0
    rs = residual_stats(
        np.asarray(ref[:n], dtype=np.float32),
        np.asarray(deg[:n], dtype=np.float32),
        sr,
    )
    return float(rs.get("corr") or 0.0)


def metrics_vs_z(
    z: np.ndarray,
    z_rand: np.ndarray,
    z_comp: np.ndarray,
    *,
    s0_cos: float,
    s0_corr: float,
) -> dict:
    """以连续 z 为参照，相对 s0=sim(z,Z_orig) 计额外损失。"""
    cos_r = float(feat_cosine(z, z_rand)["cosine"])
    cos_c = float(feat_cosine(z, z_comp)["cosine"])
    corr_r = feat_corr(z, z_rand)
    corr_c = feat_corr(z, z_comp)
    t = min(z.shape[-1], z_rand.shape[-1], z_comp.shape[-1])
    flat_cos_r = float(cosine_sim(z[:, :t].reshape(-1), z_rand[:, :t].reshape(-1)))
    return {
        "cosine": dru_from_sims(cos_r, cos_c, s0_cos),
        "corr": dru_from_sims(corr_r, corr_c, s0_corr),
        "check_flat_cos_rand": flat_cos_r,
    }


def parse_layers(s: str, n_q: int) -> list[int]:
    s = (s or "").strip()
    if not s or s.lower() in ("all", "*"):
        return list(range(int(n_q)))
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    out = sorted({i for i in out if 0 <= i < int(n_q)})
    if not out:
        raise SystemExit(f"empty --layers after parse (n_q={n_q})")
    return out


def list_wavs(wav_dir: Path, n_files: int) -> list[Path]:
    files = sorted(wav_dir.glob("*.wav"))
    if not files:
        raise SystemExit(f"no wav in {wav_dir}")
    if int(n_files) > 0:
        files = files[: int(n_files)]
    return files


def summarize(rows: list[dict], layers: list[int]) -> dict:
    by_layer: dict = {}
    sim_tok_cos, sim_tok_corr, wav_c0 = [], [], []
    for row in rows:
        a = row.get("anchor_vs_z") or {}
        if a.get("cos_tokens") is not None:
            sim_tok_cos.append(float(a["cos_tokens"]))
        if a.get("corr_tokens") is not None:
            sim_tok_corr.append(float(a["corr_tokens"]))
        if a.get("wav_corr_recon") is not None:
            wav_c0.append(float(a["wav_corr_recon"]))
    for L in layers:
        buckets = {
            "cos": {"sr": [], "sc": [], "d": [], "r": [], "u": []},
            "corr": {"sr": [], "sc": [], "d": [], "r": [], "u": []},
            "wav": {"sr": [], "sc": [], "d": [], "r": [], "u": []},
        }
        frac_post = []
        for row in rows:
            blk = (row.get("by_layer") or {}).get(str(L))
            if not blk:
                continue
            for key, name in (("cos", "cosine"), ("corr", "corr"), ("wav", "wav_corr")):
                m = blk.get(name) or {}
                if m.get("sim_rand") is None:
                    continue
                b = buckets[key]
                b["sr"].append(float(m["sim_rand"]))
                b["sc"].append(float(m["sim_comp"]))
                b["d"].append(float(m["D"]))
                if m.get("denom_ok"):
                    b["r"].append(float(m["R"]))
                    b["u"].append(float(m["U"]))
            meta = blk.get("compensation") or {}
            if meta.get("mean_frac_diff_post") is not None:
                frac_post.append(float(meta["mean_frac_diff_post"]))

        def pack(b: dict) -> dict:
            return {
                "mean_sim_rand": _mean(b["sr"]),
                "mean_sim_comp": _mean(b["sc"]),
                "mean_D": _mean(b["d"]),
                "std_D": _std(b["d"]),
                "mean_R": _mean(b["r"]),
                "std_R": _std(b["r"]),
                "mean_U": _mean(b["u"]),
                "std_U": _std(b["u"]),
                "n_denom_ok": len(b["r"]),
            }

        by_layer[str(L)] = {
            "layer": int(L),
            "n": len(buckets["cos"]["sr"]),
            "cosine": pack(buckets["cos"]),
            "corr": pack(buckets["corr"]),
            "wav_corr": pack(buckets["wav"]),
            "mean_frac_diff_post": _mean(frac_post),
        }
    return {
        "n_clips": len(rows),
        "layers": list(layers),
        "by_layer": by_layer,
        "anchor_vs_z": {
            "mean_cos_tokens": _mean(sim_tok_cos),
            "mean_corr_tokens": _mean(sim_tok_corr),
            "mean_wav_corr_recon": _mean(wav_c0),
            "note": "per-clip s0=sim(z,Z_orig), c0=corr(x,x_recon); then mean of per-clip D/R/U",
        },
        "protocol": {
            "baseline": "continuous z; extra-loss vs normal quantization s0",
            "s0": "sim(z, Z_orig) per clip (cos/corr)",
            "c0": "corr(x, x_recon) per clip",
            "Z": "sum z_q over all RVQ layers from codes",
            "rand": "single-layer random replace + freeze post layers",
            "comp": "single-layer random replace + peel z_q(L*) + requantize post",
            "D": "s0 - sim(z, Z_rand)  # Layer Impact (extra loss)",
            "R": "(sim_comp - sim_rand)/D  # Replaceability",
            "U": "(s0 - sim_comp)/D  # Irreplaceability",
            "identity": "R + U = 1 when D > eps",
            "aggregation": "per-clip ratios then mean/std",
            "eps_D": EPS_D,
        },
    }


def write_findings(path: Path, summary: dict) -> None:
    n = summary["n_clips"]
    a = summary.get("anchor_vs_z") or {}
    lines = [
        f"# Layer Impact / Replaceability / Irreplaceability（n={n}，额外损失相对 s0）\n\n",
        "单层随机替换；`Z = sum_i z_q^{(i)}`；参照连续 `z`；"
        "**正常量化基线**（逐样本）`s0 = sim(z, Z_orig)`，不是 1。\n\n",
        f"- 均值 `s0`：cos={_fmt(a.get('mean_cos_tokens'))}，"
        f"corr={_fmt(a.get('mean_corr_tokens'))}；"
        f"`c0=corr(x,x_recon)`={_fmt(a.get('mean_wav_corr_recon'))}\n",
        "- **Layer Impact** `D = s0 - sim(z, Z_rand)`\n",
        "- **Replaceability** `R = (sim_comp - sim_rand) / D`\n",
        "- **Irreplaceability** `U = (s0 - sim_comp) / D`\n",
        f"- `D < {EPS_D:g}` 时不报告 R/U（分母过小）；先看 D 再谈可替代性\n",
        "- 聚合：先逐样本算 D/R/U，再 mean（±std）\n\n",
        "## Cosine 主表（vs z，相对 s0）\n\n",
        "| L | cos(z,rand) | cos(z,comp) | D | R | U | n_ok | frac_post |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for L in summary["layers"]:
        b = summary["by_layer"][str(L)]
        c = b["cosine"]
        if c.get("std_R") is not None and c["mean_R"] is not None:
            r_s = f"{c['mean_R']:.4f}±{c['std_R']:.4f}"
        else:
            r_s = _fmt(c["mean_R"])
        if c.get("std_U") is not None and c["mean_U"] is not None:
            u_s = f"{c['mean_U']:.4f}±{c['std_U']:.4f}"
        else:
            u_s = _fmt(c["mean_U"])
        lines.append(
            f"| {L} | {_fmt(c['mean_sim_rand'])} | {_fmt(c['mean_sim_comp'])} | "
            f"{_fmt(c['mean_D'])} | {r_s} | {u_s} | {c.get('n_denom_ok', '—')} | "
            f"{_fmt(b['mean_frac_diff_post'])} |\n"
        )
    lines.extend(
        [
            "\n## Feat-Corr 主表（vs z，相对 s0）\n\n",
            "| L | corr(z,rand) | corr(z,comp) | D | R | U | n_ok |\n",
            "|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for L in summary["layers"]:
        b = summary["by_layer"][str(L)]
        q = b["corr"]
        lines.append(
            f"| {L} | {_fmt(q['mean_sim_rand'])} | {_fmt(q['mean_sim_comp'])} | "
            f"{_fmt(q['mean_D'])} | {_fmt(q['mean_R'])} | {_fmt(q['mean_U'])} | "
            f"{q.get('n_denom_ok', '—')} |\n"
        )
    lines.extend(
        [
            "\n## 波形 Corr 主表（相对 c0=corr(x,x_recon)）\n\n",
            "| L | corr(wav,rand) | corr(wav,comp) | D | R | U | n_ok |\n",
            "|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for L in summary["layers"]:
        b = summary["by_layer"][str(L)]
        w = b.get("wav_corr") or {}
        lines.append(
            f"| {L} | {_fmt(w.get('mean_sim_rand'))} | {_fmt(w.get('mean_sim_comp'))} | "
            f"{_fmt(w.get('mean_D'))} | {_fmt(w.get('mean_R'))} | {_fmt(w.get('mean_U'))} | "
            f"{w.get('n_denom_ok', '—')} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def write_layer_csv(path: Path, summary: dict) -> None:
    fields = [
        "layer",
        "n",
        "cos_sim_rand",
        "cos_sim_comp",
        "cos_D",
        "cos_R",
        "cos_U",
        "cos_n_ok",
        "corr_sim_rand",
        "corr_sim_comp",
        "corr_D",
        "corr_R",
        "corr_U",
        "wav_corr_rand",
        "wav_corr_comp",
        "wav_D",
        "wav_R",
        "wav_U",
        "wav_n_ok",
        "frac_diff_post",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for L in summary["layers"]:
            b = summary["by_layer"][str(L)]
            c, q = b["cosine"], b["corr"]
            wv = b.get("wav_corr") or {}
            w.writerow(
                {
                    "layer": L,
                    "n": b["n"],
                    "cos_sim_rand": c["mean_sim_rand"],
                    "cos_sim_comp": c["mean_sim_comp"],
                    "cos_D": c["mean_D"],
                    "cos_R": c["mean_R"],
                    "cos_U": c["mean_U"],
                    "cos_n_ok": c.get("n_denom_ok"),
                    "corr_sim_rand": q["mean_sim_rand"],
                    "corr_sim_comp": q["mean_sim_comp"],
                    "corr_D": q["mean_D"],
                    "corr_R": q["mean_R"],
                    "corr_U": q["mean_U"],
                    "wav_corr_rand": wv.get("mean_sim_rand"),
                    "wav_corr_comp": wv.get("mean_sim_comp"),
                    "wav_D": wv.get("mean_D"),
                    "wav_R": wv.get("mean_R"),
                    "wav_U": wv.get("mean_U"),
                    "wav_n_ok": wv.get("n_denom_ok"),
                    "frac_diff_post": b["mean_frac_diff_post"],
                }
            )


def recompute_from_per_clip(out_dir: Path) -> dict:
    """用已有 sim 与逐样本 s0/c0 重算 D/R/U，无需 GPU。"""
    per_dir = out_dir / "per_clip"
    rows = []
    layers_set: set[int] = set()
    for p in sorted(per_dir.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        a = rec.get("anchor_vs_z") or {}
        s0_cos = float(a["cos_tokens"])
        s0_corr = float(a.get("corr_tokens", s0_cos))
        c0 = float(a["wav_corr_recon"])
        for Lk, blk in (rec.get("by_layer") or {}).items():
            layers_set.add(int(Lk))
            for key, s0 in (
                ("cosine", s0_cos),
                ("corr", s0_corr),
                ("wav_corr", c0),
            ):
                m = blk.get(key) or {}
                if "sim_rand" not in m:
                    continue
                blk[key] = dru_from_sims(m["sim_rand"], m["sim_comp"], s0)
        rec["baseline"] = "continuous_z_extra_loss_vs_s0"
        dump_json(p, rec)
        rows.append(rec)
    layers = sorted(layers_set)
    summary = summarize(rows, layers)
    dump_json(
        out_dir / "summary.json",
        {
            **summary,
            "recomputed": True,
            "baseline_fix": "s0_per_clip",
            "clips": [r["name"] for r in rows],
        },
    )
    write_layer_csv(out_dir / "by_layer.csv", summary)
    write_findings(out_dir / "FINDINGS.md", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="单层随机替换（相对 s0 额外损失）：Layer Impact / Replaceability / Irreplaceability"
    )
    add_model_args(p, n_files=0)
    p.add_argument("--layers", type=str, default="all", help="如 all 或 0-15,9 或 0,1,2,9")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--codebook-size", type=int, default=0, help="0=从模型读")
    p.add_argument(
        "--recompute-only",
        action="store_true",
        help="仅用已有 per_clip 的 sim 与 s0/c0 重算 D/R/U",
    )
    args = p.parse_args()

    if not str(args.wav_dir or "").strip() or args.wav_dir == str(
        Path(r"/home/ma-user/work/ynian/codec_an/exp_watermark/data/mix_1500")
    ):
        args.wav_dir = str(DEFAULT_WAV_DIR)
    if not str(args.out_dir or "").strip():
        args.out_dir = str(DEFAULT_OUT)
    out_dir = Path(args.out_dir)

    if args.recompute_only:
        summary = recompute_from_per_clip(out_dir)
        a = summary.get("anchor_vs_z") or {}
        print(
            f"recomputed n={summary['n_clips']} s0_cos={_fmt(a.get('mean_cos_tokens'))} "
            f"c0={_fmt(a.get('mean_wav_corr_recon'))} -> {out_dir}",
            flush=True,
        )
        for L in (0, 1, 2, 9, 20):
            b = summary["by_layer"][str(L)]["cosine"]
            print(
                f"  L{L} cos D={_fmt(b['mean_D'])} R={_fmt(b['mean_R'])} U={_fmt(b['mean_U'])} "
                f"n_ok={b.get('n_denom_ok')}",
                flush=True,
            )
        return

    if not Path(str(args.model_path)).exists():
        args.model_path = default_model_path()

    set_seed(int(args.seed))
    device = str(args.device).strip() or auto_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = int(args.n_files)
    model, sr = load_model(args.model_path, device)
    n_q = int(len(model.quantizer.quantizers))
    q0 = model.quantizer.quantizers[0]
    cb_size = int(args.codebook_size)
    if cb_size <= 0:
        for attr in ("codebook_size", "n_embed", "num_embeddings"):
            v = getattr(q0, attr, None)
            if v:
                cb_size = int(v)
                break
        if cb_size <= 0:
            w = getattr(getattr(q0, "embedding", None), "weight", None)
            if w is None:
                w = getattr(getattr(q0, "codebook", None), "weight", None)
            cb_size = int(w.shape[0]) if w is not None and w.ndim >= 2 else 1024
    layers = parse_layers(args.layers, n_q)
    paths = list_wavs(Path(args.wav_dir), n_files if n_files > 0 else 10**9)

    per_dir = out_dir / "per_clip"
    per_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"layer_replaceability (extra-loss vs s0) n={len(paths)} layers={layers[0]}..{layers[-1]} "
        f"(n_layers={len(layers)}) cb={cb_size} seed={args.seed} -> {out_dir}",
        flush=True,
    )

    rows = []
    for i, path in enumerate(paths):
        name = path.stem
        print(f"clip {i+1}/{len(paths)} {name}", flush=True)
        wav = np.asarray(read_wav(path, sr), dtype=np.float32).reshape(-1)
        z = continuous_z(model, wav)
        codes_orig = encode_moss(model, wav)
        z_tokens = full_zq_sum(model, codes_orig)
        t = min(z.shape[-1], z_tokens.shape[-1])
        z = z[:, :t]
        z_tokens = z_tokens[:, :t]
        s0_cos = float(feat_cosine(z, z_tokens)["cosine"])
        s0_corr = feat_corr(z, z_tokens)
        wav_recon = decode_codes(model, wav, codes_orig)
        c0 = wav_corr(wav, wav_recon, sr)
        anchor = {
            "cos_tokens": s0_cos,
            "corr_tokens": s0_corr,
            "wav_corr_recon": c0,
            "T": int(t),
        }
        print(
            f"  s0 cos/corr={s0_cos:.4f}/{s0_corr:.4f}  c0 wav={c0:.4f}",
            flush=True,
        )

        by_layer = {}
        for L in layers:
            rng = np.random.default_rng(int(args.seed) + 10007 * (i + 1) + 17 * int(L))
            codes_rand = random_replace_layer(codes_orig, L, cb_size, rng)
            z_rand = full_zq_sum(model, codes_rand)
            if int(L) >= n_q - 1:
                codes_comp = codes_rand
                meta = {
                    "t_use": int(codes_orig.shape[1]),
                    "wm_layer": int(L),
                    "subtract_wm_layer": True,
                    "post_layers_changed": {},
                    "mean_frac_diff_post": 0.0,
                    "note": "last layer: no post layers",
                }
            else:
                _, codes_comp, meta = compensate_post_layers(
                    model, wav, codes_rand, int(L), subtract_wm_layer=True
                )
            z_comp = full_zq_sum(model, codes_comp)
            m = metrics_vs_z(z, z_rand, z_comp, s0_cos=s0_cos, s0_corr=s0_corr)
            wav_rand = decode_codes(model, wav, codes_rand)
            wav_comp = decode_codes(model, wav, codes_comp)
            w_dru = dru_from_sims(wav_corr(wav, wav_rand, sr), wav_corr(wav, wav_comp, sr), c0)
            blk = {
                "layer": int(L),
                "n_changed": int(np.sum(codes_rand[L] != codes_orig[L])),
                "T": int(codes_orig.shape[1]),
                "cosine": m["cosine"],
                "corr": m["corr"],
                "wav_corr": w_dru,
                "compensation": {
                    "mean_frac_diff_post": meta.get("mean_frac_diff_post"),
                    "post_layers_changed": {
                        str(k): v for k, v in (meta.get("post_layers_changed") or {}).items()
                    },
                },
            }
            by_layer[str(L)] = blk
            print(
                f"  L{L:02d} cos D={m['cosine']['D']:.4f} R={_fmt(m['cosine']['R'])} "
                f"U={_fmt(m['cosine']['U'])} | wav D={w_dru['D']:.4f} R={_fmt(w_dru['R'])}",
                flush=True,
            )
        rec = {
            "name": name,
            "path": str(path),
            "n_q": n_q,
            "codebook_size": cb_size,
            "baseline": "continuous_z_extra_loss_vs_s0",
            "anchor_vs_z": anchor,
            "by_layer": by_layer,
        }
        dump_json(per_dir / f"{name}.json", rec)
        rows.append(rec)

    summary = summarize(rows, layers)
    dump_json(
        out_dir / "summary.json",
        {
            **summary,
            "seed": int(args.seed),
            "codebook_size": cb_size,
            "n_q": n_q,
            "wav_dir": str(args.wav_dir),
            "clips": [r["name"] for r in rows],
        },
    )
    write_layer_csv(out_dir / "by_layer.csv", summary)
    write_findings(out_dir / "FINDINGS.md", summary)
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
