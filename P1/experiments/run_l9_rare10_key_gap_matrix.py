"""L9 · 10 ABA key 闭集溯源：rare vs random 选 key 对照。

选 key
------
- ``rare``：稀有簇升序，相邻两簇最稀 token → ABA（互不重用簇）
- ``random``：码本均匀随机抽 2N 个 distinct token 配对 ABA
- ``both``（默认）：两套都跑，COMPARE 看 rare 是否更好

协议
----
嵌 key_i → ``score_wav_once``：主指标盲溯源 ``cosine=aba_embed``；``gap`` 对照。
每 clip：``{mode}/per_clip/{stem}.json`` 存 cos + gap。

例::

  python -m P1.experiments.run_l9_rare10_key_gap_matrix \\
    --key-mode both --wav-dir /path/to/mix_1500 --out-dir /path/to/out \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from base.channels import P1_BAND_L9, profile_score_cfg, score_wav_once
from base.config import (
    EXP_ROOT,
    default_cluster_dir,
    default_mix_1500,
    default_model_path,
    default_table_path,
)
from base.moss_io import read_wav, set_seed
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
from base.rare_tokens import (
    _rarest_token,
    cluster_freq_from_token_freq,
    cluster_members,
    load_layer_mapping,
    token_freq_from_trans,
)
from base.util import dump_json, embed_wav, load_model
from P1.experiments.run_anchor_attack_three_n1500 import list_clips, load_clips_from_manifest

OUT_STEM = "l9_rare10_key_gap_matrix"
_OUT_CANDS = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
    Path(r"E:\code\water_marks\result") / OUT_STEM,
    EXP_ROOT / OUT_STEM,
)
DEFAULT_OUT = next((p for p in _OUT_CANDS if p.parent.exists()), _OUT_CANDS[0])
LAYER = 9
DEFAULT_N_KEYS = 10
# v2: 官方 score_wav_once（aba_embed / slide_v3），不再用简化 band_tmpl cos
METRIC_VERSION = 2

_TRANS_CANDS = (
    Path(r"d:\data\codec_an\codec_plot\codec_layers\analysis\trans_count.npy"),
    Path(r"D:\data\codec_an\codec_plot\codec_layers\analysis\trans_count.npy"),
    Path(r"/data1/home/ynian/codes/codec_an/exp_watermark/codec_an/codec_plot/codec_layers/analysis/trans_count.npy"),
)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def resolve_trans(explicit: str = "") -> Path:
    cands = [Path(explicit)] if explicit else []
    cands.extend(_TRANS_CANDS)
    for p in cands:
        if p and p.is_file():
            return p
    raise SystemExit("cannot find trans_count.npy; set --trans")


def key_to_table(key: dict, *, layer: int = LAYER) -> dict:
    """单层 L9 ABA table，供 embed_wav / score_wav_once 共用。"""
    L = str(int(layer))
    return {
        "note": f"L{layer} rare key {key.get('key_id')}",
        "n_layers": int(layer) + 1,
        "layers": {
            L: {
                "aba": [int(x) for x in key["aba"]],
                "clusters": [int(x) for x in key["clusters"]],
                "token_freq": list(key.get("token_freq") or []),
                "cluster_freq": list(key.get("cluster_freq") or []),
            }
        },
    }


def _freq_meta(tok: int, freq: np.ndarray, mapping: dict[int, int], cf: dict[int, int]) -> dict:
    n = int(freq.shape[0])
    cid = int(mapping.get(int(tok), -1))
    return {
        "token": int(tok),
        "cluster": cid,
        "token_freq": int(freq[tok]) if 0 <= int(tok) < n else 0,
        "cluster_freq": int(cf.get(cid, 0)) if cid >= 0 else 0,
    }


def build_l9_rare_keys(
    n_keys: int = DEFAULT_N_KEYS,
    *,
    layer: int = LAYER,
    cluster_dir: str = "",
    trans_path: str = "",
) -> list[dict]:
    """稀有簇升序，相邻两簇各取最稀 token → ABA key（互不重用簇）。"""
    cd = cluster_dir or default_cluster_dir()
    trans = np.load(str(resolve_trans(trans_path)))
    freq = token_freq_from_trans(trans)[int(layer)]
    mapping = load_layer_mapping(int(layer), cd)
    members = cluster_members(mapping)
    cf = cluster_freq_from_token_freq(freq, mapping)
    ranked = []
    for cid, mem in members.items():
        tok = _rarest_token(mem, freq, set())
        if tok is None:
            continue
        ranked.append(
            {
                "cluster": int(cid),
                "token": int(tok),
                "cluster_freq": int(cf.get(cid, 0)),
                "token_freq": int(freq[tok]) if 0 <= tok < freq.shape[0] else 0,
            }
        )
    ranked.sort(key=lambda r: (r["cluster_freq"], r["token_freq"], r["cluster"], r["token"]))
    need = int(n_keys) * 2
    if len(ranked) < need:
        raise SystemExit(f"L{layer} only {len(ranked)} usable clusters; need {need} for {n_keys} keys")
    keys = []
    for i in range(int(n_keys)):
        x, y = ranked[2 * i], ranked[2 * i + 1]
        a, b = int(x["token"]), int(y["token"])
        if a == b:
            raise SystemExit(f"degenerate ABA at key {i}: a==b=={a}")
        keys.append(
            {
                "key_id": int(i),
                "a": a,
                "b": b,
                "aba": [a, b, a],
                "clusters": [int(x["cluster"]), int(y["cluster"]), int(x["cluster"])],
                "token_freq": [int(x["token_freq"]), int(y["token_freq"])],
                "cluster_freq": [int(x["cluster_freq"]), int(y["cluster_freq"])],
                "layer": int(layer),
                "mode": "rare",
            }
        )
    return keys


def build_l9_random_keys(
    n_keys: int = DEFAULT_N_KEYS,
    *,
    layer: int = LAYER,
    seed: int = 0,
    codebook_size: int = 1024,
    cluster_dir: str = "",
    trans_path: str = "",
) -> list[dict]:
    """码本均匀随机抽 2N 个 distinct token，顺序配对 ABA。"""
    need = int(n_keys) * 2
    k = int(codebook_size)
    if need > k:
        raise SystemExit(f"need {need} tokens > codebook {k}")
    rng = np.random.default_rng(int(seed))
    toks = rng.choice(k, size=need, replace=False).astype(np.int64)

    cd = cluster_dir or default_cluster_dir()
    mapping = load_layer_mapping(int(layer), cd)
    try:
        trans = np.load(str(resolve_trans(trans_path)))
        freq = token_freq_from_trans(trans)[int(layer)]
        cf = cluster_freq_from_token_freq(freq, mapping)
    except SystemExit:
        freq = np.zeros(k, dtype=np.int64)
        cf = {}

    keys = []
    for i in range(int(n_keys)):
        xa = _freq_meta(int(toks[2 * i]), freq, mapping, cf)
        xb = _freq_meta(int(toks[2 * i + 1]), freq, mapping, cf)
        a, b = int(xa["token"]), int(xb["token"])
        keys.append(
            {
                "key_id": int(i),
                "a": a,
                "b": b,
                "aba": [a, b, a],
                "clusters": [int(xa["cluster"]), int(xb["cluster"]), int(xa["cluster"])],
                "token_freq": [int(xa["token_freq"]), int(xb["token_freq"])],
                "cluster_freq": [int(xa["cluster_freq"]), int(xb["cluster_freq"])],
                "layer": int(layer),
                "mode": "random",
                "seed": int(seed),
            }
        )
    return keys


def build_keys_for_mode(mode: str, args) -> list[dict]:
    mode = str(mode).strip().lower()
    if mode == "rare":
        return build_l9_rare_keys(
            int(args.n_keys),
            layer=int(args.layer),
            cluster_dir=str(args.cluster_dir),
            trans_path=str(args.trans),
        )
    if mode == "random":
        return build_l9_random_keys(
            int(args.n_keys),
            layer=int(args.layer),
            seed=int(args.seed),
            cluster_dir=str(args.cluster_dir),
            trans_path=str(args.trans),
        )
    raise SystemExit(f"unknown key mode: {mode}")


def _mat_ok(m, n_keys: int) -> bool:
    return (
        isinstance(m, list)
        and len(m) == n_keys
        and all(isinstance(row, list) and len(row) == n_keys for row in m)
    )


def clip_ok(path: Path, n_keys: int) -> bool:
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(doc, dict) or doc.get("metric_version") != METRIC_VERSION or "error" in doc:
        return False
    return _mat_ok(doc.get("cosine"), n_keys) and _mat_ok(doc.get("gap"), n_keys)


def _score(model, wav, sr, table) -> dict:
    cfg = profile_score_cfg(table, P1_BAND_L9)
    return score_wav_once(model, wav, sr, cfg, n_peel=LAYER, period=3)


def _worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    device = str(payload["device"])
    pin_npu(device)
    log_fp, log = open_worker_log(payload)
    per_dir = Path(payload["per_clip_dir"])
    per_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(payload.get("resume"))
    keys = list(payload["keys"])
    n_keys = len(keys)
    layer = int(payload.get("layer") or LAYER)
    n_ok = n_skip = n_err = 0

    try:
        set_seed(int(payload.get("seed") or 0) + int(payload["worker_id"]))
        model, sr = load_model(str(payload["moss_path"]), device)
        tables = [key_to_table(k, layer=layer) for k in keys]
        log(f"start clips={len(payload['items'])} n_keys={n_keys} L{layer} score=score_wav_once")

        for i, (stem, path) in enumerate(payload["items"]):
            safe = str(stem).replace("/", "_").replace("\\", "_")
            out_json = per_dir / f"{safe}.json"
            if resume and clip_ok(out_json, n_keys):
                n_skip += 1
                n_ok += 1
                continue
            try:
                wav = np.asarray(read_wav(path, int(sr)), dtype=np.float32).reshape(-1)

                # 干净语音对各 key 的官方检测分（gap 分子）
                orig_aba = np.zeros(n_keys, dtype=np.float64)
                orig_slide = np.zeros(n_keys, dtype=np.float64)
                for j, tab in enumerate(tables):
                    so = _score(model, wav, int(sr), tab)
                    orig_aba[j] = float(so["aba_embed"])
                    orig_slide[j] = float(so["slide18"])

                cos_m = np.zeros((n_keys, n_keys), dtype=np.float64)  # aba_embed(wm, key_j)
                gap_m = np.zeros((n_keys, n_keys), dtype=np.float64)
                slide_m = np.zeros((n_keys, n_keys), dtype=np.float64)
                gap_slide_m = np.zeros((n_keys, n_keys), dtype=np.float64)
                hit_gap = hit_cos = hit_slide = hit_gap_slide = 0
                n_changed = []

                for ki, tab_i in enumerate(tables):
                    emb = embed_wav(
                        model, wav, tab_i, high_start=layer, high_end=layer, layers=[layer], period=3
                    )
                    wm = np.asarray(emb["wav"], dtype=np.float32).reshape(-1)
                    n_changed.append(int(emb.get("n_changed") or 0))
                    for kj, tab_j in enumerate(tables):
                        sw = _score(model, wm, int(sr), tab_j)
                        aba = float(sw["aba_embed"])
                        sl = float(sw["slide18"])
                        cos_m[ki, kj] = aba
                        gap_m[ki, kj] = aba - float(orig_aba[kj])
                        slide_m[ki, kj] = sl
                        gap_slide_m[ki, kj] = sl - float(orig_slide[kj])
                    if int(np.argmax(gap_m[ki])) == ki:
                        hit_gap += 1
                    if int(np.argmax(cos_m[ki])) == ki:
                        hit_cos += 1
                    if int(np.argmax(slide_m[ki])) == ki:
                        hit_slide += 1
                    if int(np.argmax(gap_slide_m[ki])) == ki:
                        hit_gap_slide += 1

                _write_json(
                    out_json,
                    {
                        "stem": stem,
                        "src": path,
                        "layer": layer,
                        "n_keys": n_keys,
                        "metric_version": METRIC_VERSION,
                        "score": "score_wav_once / aba_embed + slide_v3",
                        "sr": int(sr),
                        "n_changed": n_changed,
                        # 干净语音对各检测 key
                        "orig_cosine": orig_aba.tolist(),
                        "orig_slide": orig_slide.tolist(),
                        # 行=嵌入 key i，列=检测 key j；cos 与 gap 都保存
                        "cosine": cos_m.tolist(),
                        "gap": gap_m.tolist(),
                        "slide": slide_m.tolist(),
                        "gap_slide": gap_slide_m.tolist(),
                        "argmax_hit_cosine": int(hit_cos),
                        "argmax_hit_gap": int(hit_gap),
                        "argmax_hit_slide": int(hit_slide),
                        "argmax_hit_gap_slide": int(hit_gap_slide),
                    },
                )
                n_ok += 1
            except Exception as exc:
                n_err += 1
                _write_json(
                    out_json,
                    {"stem": stem, "src": path, "error": str(exc), "trace": traceback.format_exc()},
                )
                log(f"ERR {stem}: {exc}")
            if (i + 1) % 10 == 0 or i == 0:
                log(f"progress {i+1}/{len(payload['items'])} ok={n_ok} skip={n_skip} err={n_err}")
    finally:
        free_device()
        if log_fp:
            log_fp.close()
    return {"worker_id": payload["worker_id"], "n_ok": n_ok, "n_skip": n_skip, "n_err": n_err}


def _finite_mean(mats: list[np.ndarray]) -> np.ndarray | None:
    if not mats:
        return None
    return np.nanmean(np.stack(mats, axis=0), axis=0)


def matrix_stats(mat: np.ndarray) -> dict:
    mat = np.asarray(mat, dtype=np.float64)
    n = mat.shape[0]
    diag = np.diag(mat)
    off = mat[~np.eye(n, dtype=bool)]
    margins = []
    for i in range(n):
        row = mat[i].copy()
        row[i] = -np.inf
        margins.append(float(diag[i] - np.max(row)))
    return {
        "diag_mean": float(np.mean(diag)),
        "offdiag_mean": float(np.mean(off)),
        "sep": float(np.mean(diag) - np.mean(off)),
        "row_margin_mean": float(np.mean(margins)),
        "row_margin_min": float(np.min(margins)),
        "n_keys": int(n),
    }


def aggregate(per_dir: Path, n_keys: int) -> dict:
    buckets = {"cosine": [], "gap": [], "slide": [], "gap_slide": []}
    hits = {k: 0 for k in buckets}
    n = 0
    for p in sorted(per_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("error") or d.get("metric_version") != METRIC_VERSION:
            continue
        if not _mat_ok(d.get("cosine"), n_keys) or not _mat_ok(d.get("gap"), n_keys):
            continue
        for name in buckets:
            m = d.get(name)
            if _mat_ok(m, n_keys):
                buckets[name].append(np.asarray(m, dtype=np.float64))
        hits["cosine"] += int(d.get("argmax_hit_cosine") or 0)
        hits["gap"] += int(d.get("argmax_hit_gap") or 0)
        hits["slide"] += int(d.get("argmax_hit_slide") or 0)
        hits["gap_slide"] += int(d.get("argmax_hit_gap_slide") or 0)
        n += 1

    total = n * n_keys
    out: dict = {
        "n_clips": n,
        "n_keys": n_keys,
        "metric_version": METRIC_VERSION,
        "score": "score_wav_once aba_embed/slide_v3; gap = score_wm − score_orig",
    }
    for name, mats in buckets.items():
        mean = _finite_mean(mats)
        out[f"argmax_hit_rate_{name}"] = (hits[name] / total) if total else None
        if mean is not None:
            out[f"mean_{name}"] = mean.tolist()
            out[f"{name}_stats"] = matrix_stats(mean)

    # 盲溯源主看 cos；gap 需原件，仅对照
    if "gap_stats" in out and "cosine_stats" in out:
        gs, cs = out["gap_stats"], out["cosine_stats"]
        out["cos_better_vs_gap"] = {
            "sep": bool(cs["sep"] > gs["sep"]),
            "row_margin_mean": bool(cs["row_margin_mean"] > gs["row_margin_mean"]),
            "argmax_hit": bool(
                (out.get("argmax_hit_rate_cosine") or 0) >= (out.get("argmax_hit_rate_gap") or 0)
            ),
            "note": "闭集溯源主指标=aba_embed(cos)；gap 需 orig，仅对照",
        }
    if "gap_slide_stats" in out and "slide_stats" in out:
        gss, ss = out["gap_slide_stats"], out["slide_stats"]
        out["slide_better_vs_gap_slide"] = {
            "sep": bool(ss["sep"] > gss["sep"]),
            "row_margin_mean": bool(ss["row_margin_mean"] > gss["row_margin_mean"]),
            "argmax_hit": bool(
                (out.get("argmax_hit_rate_slide") or 0)
                >= (out.get("argmax_hit_rate_gap_slide") or 0)
            ),
        }
    return out


def _md_mat(mat: list[list[float]], n: int) -> list[str]:
    hdr = "|  | " + " | ".join(f"D{j}" for j in range(n)) + " |"
    sep = "|---|" + "|".join(["---:"] * n) + "|"
    body = [
        "| E" + str(i) + " | " + " | ".join(f"{float(x):.4f}" for x in row) + " |"
        for i, row in enumerate(mat)
    ]
    return [hdr, sep, *body, ""]


def render_md(summary: dict, keys: list[dict]) -> str:
    def f(x, nd=4):
        return "—" if x is None else f"{float(x):.{nd}f}"

    def pct(x):
        return "—" if x is None else f"{100 * float(x):.2f}%"

    cs = summary.get("cosine_stats") or {}
    gs = summary.get("gap_stats") or {}
    ss = summary.get("slide_stats") or {}
    gss = summary.get("gap_slide_stats") or {}
    better = summary.get("cos_better_vs_gap") or {}
    better_s = summary.get("slide_better_vs_gap_slide") or {}
    n = int(summary.get("n_keys") or len(keys))
    lines = [
        "# L9 rare-10 key 闭集溯源（主：cos / 对照：gap）",
        "",
        f"- n_clips={summary.get('n_clips')}  n_keys={n}  layer=L{LAYER}",
        f"- score: `{summary.get('score')}`",
        f"- keys: " + ", ".join(f"K{k['key_id']}=({k['a']},{k['b']})" for k in keys),
        "- 盲溯源主看 **cosine=aba_embed**；gap 需原件，仅对照",
        "",
        "## 结论：溯源上 cos 是否优于 gap？（aba 通道）",
        "",
        "| 指标 | cosine (盲) | gap (需orig) | cos更好? |",
        "|---|---:|---:|:---:|",
        f"| sep (diag−off) | {f(cs.get('sep'))} | {f(gs.get('sep'))} | "
        f"{'Y' if better.get('sep') else 'N'} |",
        f"| row_margin_mean | {f(cs.get('row_margin_mean'))} | {f(gs.get('row_margin_mean'))} | "
        f"{'Y' if better.get('row_margin_mean') else 'N'} |",
        f"| argmax hit | {pct(summary.get('argmax_hit_rate_cosine'))} | "
        f"{pct(summary.get('argmax_hit_rate_gap'))} | "
        f"{'Y' if better.get('argmax_hit') else 'N'} |",
        "",
        "## slide 通道对照",
        "",
        "| 指标 | slide (盲) | gap_slide | slide更好? |",
        "|---|---:|---:|:---:|",
        f"| sep | {f(ss.get('sep'))} | {f(gss.get('sep'))} | "
        f"{'Y' if better_s.get('sep') else 'N'} |",
        f"| row_margin_mean | {f(ss.get('row_margin_mean'))} | {f(gss.get('row_margin_mean'))} | "
        f"{'Y' if better_s.get('row_margin_mean') else 'N'} |",
        f"| argmax hit | {pct(summary.get('argmax_hit_rate_slide'))} | "
        f"{pct(summary.get('argmax_hit_rate_gap_slide'))} | "
        f"{'Y' if better_s.get('argmax_hit') else 'N'} |",
        "",
        "## mean cosine（aba_embed，主）矩阵",
        "",
    ]
    if summary.get("mean_cosine"):
        lines += _md_mat(summary["mean_cosine"], n)
    lines += ["## mean gap（对照）矩阵", ""]
    if summary.get("mean_gap"):
        lines += _md_mat(summary["mean_gap"], n)
    lines += ["## mean slide 矩阵", ""]
    if summary.get("mean_slide"):
        lines += _md_mat(summary["mean_slide"], n)
    lines += ["## mean gap_slide 矩阵", ""]
    if summary.get("mean_gap_slide"):
        lines += _md_mat(summary["mean_gap_slide"], n)
    return "\n".join(lines)


def render_compare(rare_s: dict, rand_s: dict) -> str:
    """rare vs random：主看盲溯源 cos 的 hit / sep / margin。"""

    def f(x, nd=4):
        return "—" if x is None else f"{float(x):.{nd}f}"

    def pct(x):
        return "—" if x is None else f"{100 * float(x):.2f}%"

    def pack(s: dict, name: str) -> dict:
        cs = s.get("cosine_stats") or {}
        return {
            "n": s.get("n_clips"),
            "hit": s.get("argmax_hit_rate_cosine"),
            "sep": cs.get("sep"),
            "margin": cs.get("row_margin_mean"),
            "gap_hit": s.get("argmax_hit_rate_gap"),
            "label": name,
        }

    r, u = pack(rare_s, "rare"), pack(rand_s, "random")

    def better(a, b, higher=True):
        if a is None or b is None:
            return "—"
        ok = (float(a) >= float(b)) if higher else (float(a) <= float(b))
        return "rare" if ok else "random"

    lines = [
        "# rare vs random key 闭集溯源对照（主指标：aba_embed cos）",
        "",
        "| 指标 | rare | random | 更好 |",
        "|---|---:|---:|:---:|",
        f"| n_clips | {r['n']} | {u['n']} |  |",
        f"| cos argmax hit | {pct(r['hit'])} | {pct(u['hit'])} | {better(r['hit'], u['hit'])} |",
        f"| cos sep | {f(r['sep'])} | {f(u['sep'])} | {better(r['sep'], u['sep'])} |",
        f"| cos row_margin | {f(r['margin'])} | {f(u['margin'])} | {better(r['margin'], u['margin'])} |",
        f"| gap argmax hit（对照） | {pct(r['gap_hit'])} | {pct(u['gap_hit'])} | {better(r['gap_hit'], u['gap_hit'])} |",
        "",
        "结论看 **cos argmax hit / sep / margin**：rare 是否整体优于 random。",
        "",
    ]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="L9 ABA key 闭集溯源：rare vs random")
    p.add_argument("--wav-dir", type=str, default="")
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--names", type=str, default="")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--table", type=str, default=default_table_path(), help="仅作路径参考")
    p.add_argument("--cluster-dir", type=str, default=default_cluster_dir())
    p.add_argument("--trans", type=str, default="")
    p.add_argument("--n-keys", type=int, default=DEFAULT_N_KEYS)
    p.add_argument(
        "--key-mode",
        type=str,
        default="both",
        choices=("rare", "random", "both"),
        help="选 key：rare / random / both（默认对照）",
    )
    p.add_argument("--rebuild-keys", action="store_true", help="强制重算 keybank")
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="random 模式采样种子")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    add_npu_args(p, default_workers=8, default_device="npu:0")
    return p.parse_args()


def _ensure_keybank(mode_dir: Path, mode: str, args) -> list[dict]:
    keys_path = mode_dir / "keybank.json"
    if keys_path.is_file() and not bool(args.rebuild_keys):
        doc = json.loads(keys_path.read_text(encoding="utf-8"))
        keys = doc.get("keys") or []
        if not keys:
            raise SystemExit(f"empty keys in {keys_path}")
        print(f"[{mode}] reuse keybank n={len(keys)} <- {keys_path}", flush=True)
        return keys
    keys = build_keys_for_mode(mode, args)
    dump_json(
        keys_path,
        {
            "mode": mode,
            "layer": int(args.layer),
            "n_keys": len(keys),
            "seed": int(args.seed) if mode == "random" else None,
            "cluster_dir": str(args.cluster_dir),
            "trans": str(resolve_trans(str(args.trans))),
            "selection": (
                "rarest clusters paired in order"
                if mode == "rare"
                else f"uniform random distinct tokens seed={args.seed}"
            ),
            "keys": keys,
            "tables": [key_to_table(k, layer=int(args.layer)) for k in keys],
        },
    )
    print(f"[{mode}] keybank n={len(keys)} -> {keys_path}", flush=True)
    for k in keys:
        print(
            f"  [{mode}] K{k['key_id']}: ABA=({k['a']},{k['b']}) "
            f"clusters={k['clusters'][:2]} tok_freq={k['token_freq']}",
            flush=True,
        )
    return keys


def run_one_mode(mode: str, mode_dir: Path, clips: list, src_desc: str, args) -> dict:
    mode_dir.mkdir(parents=True, exist_ok=True)
    per_dir = mode_dir / "per_clip"
    per_dir.mkdir(parents=True, exist_ok=True)
    keys = _ensure_keybank(mode_dir, mode, args)

    if bool(args.summary_only):
        summary = aggregate(per_dir, len(keys))
        dump_json(mode_dir / "summary.json", summary)
        (mode_dir / "FINDINGS.md").write_text(render_md(summary, keys), encoding="utf-8")
        print(f"[{mode}] summary-only n={summary.get('n_clips')} -> {mode_dir}", flush=True)
        return summary

    n_workers = max(1, int(args.num_workers))
    logs = mode_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (mode_dir / "RUN.md").write_text(
        f"# L9 key 溯源 · mode={mode}\n\n"
        f"- source: `{src_desc}` n={len(clips)}\n"
        f"- layer: L{args.layer}  n_keys={len(keys)}  key_mode={mode}\n"
        f"- 主指标: 盲溯源 cos=aba_embed；gap 对照\n"
        f"- workers: {n_workers}\n",
        encoding="utf-8",
    )
    print(
        f"[{mode}] n={len(clips)} keys={len(keys)} workers={n_workers} -> {mode_dir}",
        flush=True,
    )
    payloads = build_payloads(
        clips,
        num_workers=n_workers,
        npu_ids=parse_npu_ids(args.npu_ids),
        device_single=str(args.device),
        logs_dir=logs,
        common={
            "moss_path": args.moss_path,
            "per_clip_dir": str(per_dir),
            "cpu_threads": int(args.cpu_threads),
            "resume": bool(args.resume),
            "seed": int(args.seed),
            "keys": keys,
            "layer": int(args.layer),
        },
    )
    outs = run_shards(_worker, payloads, num_workers=n_workers)
    dump_json(mode_dir / "workers.json", outs)
    summary = aggregate(per_dir, len(keys))
    summary["key_mode"] = mode
    dump_json(mode_dir / "summary.json", summary)
    (mode_dir / "FINDINGS.md").write_text(render_md(summary, keys), encoding="utf-8")
    print(
        f"[{mode}] done n={summary.get('n_clips')} "
        f"cos_hit={summary.get('argmax_hit_rate_cosine')} "
        f"gap_hit={summary.get('argmax_hit_rate_gap')} -> {mode_dir}",
        flush=True,
    )
    return summary


def main() -> None:
    ensure_spawn()
    args = parse_args()
    if args.no_resume:
        args.resume = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_arg = str(args.key_mode).strip().lower()
    modes = ["rare", "random"] if mode_arg == "both" else [mode_arg]

    if bool(args.summary_only):
        summaries = {}
        for mode in modes:
            mode_dir = out_dir / mode if mode_arg == "both" else out_dir
            keys = _ensure_keybank(mode_dir, mode, args)
            summary = aggregate(mode_dir / "per_clip", len(keys))
            summary["key_mode"] = mode
            dump_json(mode_dir / "summary.json", summary)
            (mode_dir / "FINDINGS.md").write_text(render_md(summary, keys), encoding="utf-8")
            summaries[mode] = summary
            print(f"[{mode}] summary-only n={summary.get('n_clips')}", flush=True)
        if mode_arg == "both" and "rare" in summaries and "random" in summaries:
            cmp = render_compare(summaries["rare"], summaries["random"])
            (out_dir / "COMPARE_rare_vs_random.md").write_text(cmp, encoding="utf-8")
            dump_json(
                out_dir / "compare_rare_vs_random.json",
                {"rare": summaries["rare"], "random": summaries["random"]},
            )
            print(f"COMPARE -> {out_dir / 'COMPARE_rare_vs_random.md'}", flush=True)
        return

    if str(args.manifest or "").strip():
        clips = load_clips_from_manifest(Path(args.manifest), limit=int(args.limit))
        src_desc = f"manifest={args.manifest}"
    else:
        wav_dir = str(args.wav_dir or "").strip() or default_mix_1500()
        name_set = {x.strip() for x in str(args.names).replace(" ", ",").split(",") if x.strip()} or None
        clips = list_clips(Path(wav_dir), limit=int(args.limit), names=name_set)
        src_desc = f"wav_dir={wav_dir}"
    if not clips:
        raise SystemExit(f"no clips ({src_desc})")

    (out_dir / "RUN.md").write_text(
        "# L9 key 闭集溯源 rare vs random\n\n"
        f"- source: `{src_desc}` n={len(clips)}\n"
        f"- key_mode: `{mode_arg}`  n_keys={args.n_keys}  seed={args.seed}\n"
        f"- 主指标: cos=aba_embed；gap 对照\n",
        encoding="utf-8",
    )

    summaries = {}
    for mode in modes:
        mode_dir = out_dir / mode if mode_arg == "both" else out_dir
        summaries[mode] = run_one_mode(mode, mode_dir, clips, src_desc, args)

    if mode_arg == "both" and "rare" in summaries and "random" in summaries:
        cmp = render_compare(summaries["rare"], summaries["random"])
        (out_dir / "COMPARE_rare_vs_random.md").write_text(cmp, encoding="utf-8")
        dump_json(
            out_dir / "compare_rare_vs_random.json",
            {"rare": summaries["rare"], "random": summaries["random"]},
        )
        print(f"COMPARE -> {out_dir / 'COMPARE_rare_vs_random.md'}", flush=True)


if __name__ == "__main__":
    main()
