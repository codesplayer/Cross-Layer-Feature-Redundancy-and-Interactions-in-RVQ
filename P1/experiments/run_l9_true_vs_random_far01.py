"""L9 官方 key vs 随机 ABA：1500 定标 cos 阈值 → 2040 评测。

必填路径（三个）
--------------
- ``--wav-1500``：定标语音目录（mix_1500）
- ``--wav-2040``：评测语音目录（fleurs_2040）
- ``--out-dir``：结果根目录

每条语音单独保存::

  {out}/calib/per_clip/{stem}.json
  {out}/eval/per_clip/{stem}.json

协议
----
1. 真 key = ``rare_aba.json`` 的 L9 ABA（全库共用）
2. 随机 key：**每 20 条语音换一个**随机 ABA（避开真 key token；组间尽量不重复）
   1500 / 2040 各自按排序后的列表分组，因此两边都有多种随机 key
3. 每条：嵌真 key / 嵌该组随机 key；一律用**真 key** 打 ``aba_embed`` / ``slide_v3``
4. 1500：正=true_wm、负=random_wm → FAR01 定 ``T``
5. 2040：阈值冻结，报 FAR / TPR / AUC

例::

  python -m P1.experiments.run_l9_true_vs_random_far01 \\
    --wav-1500 /path/mix_1500 \\
    --wav-2040 /path/fleurs_2040 \\
    --out-dir /path/out \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from base.channels import P1_BAND_L9, profile_score_cfg, score_wav_once
from base.config import EXP_ROOT, default_model_path, default_table_path
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
from base.rare_tokens import ensure_rare_table, layer_aba
from base.util import dump_json, embed_wav, load_model
from P1.experiments.compute_layer_gap_auc import auc_mw, roc_threshold_at_far
from P1.experiments.eval_fleurs_far01_from_anchor_clean import rate_pack
from P1.experiments.run_anchor_attack_three_n1500 import list_clips

OUT_STEM = "l9_true_vs_random_far01"
DEFAULT_OUT = EXP_ROOT / OUT_STEM
LAYER = 9
TARGET_FAR = 0.01
GROUP_SIZE = 20  # 每 N 条语音共用一个随机 ABA
METRIC_VERSION = 2
CODEBOOK = 1024


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def true_l9_from_table(table: dict) -> dict:
    picked = layer_aba(table, LAYER)
    if not picked:
        raise SystemExit(f"rare table missing L{LAYER} ABA")
    aba, clusters = picked
    item = ((table.get("layers") or {}).get(str(LAYER)) or {})
    return {
        "aba": [int(x) for x in aba],
        "clusters": [int(x) for x in clusters] if clusters else [-1, -1, -1],
        "token_freq": list(item.get("token_freq") or []),
        "cluster_freq": list(item.get("cluster_freq") or []),
        "a": int(aba[0]),
        "b": int(aba[1]),
        "source": "rare_aba.json L9",
    }


def _draw_random_aba(
    rng: np.random.Generator,
    *,
    ban_tokens: set[int],
    used_pairs: set[tuple[int, int]],
    codebook: int = CODEBOOK,
) -> tuple[int, int]:
    """抽 (a,b)，避开真 key token，且 (a,b)/(b,a) 不与已用组重复。"""
    pool = [i for i in range(int(codebook)) if i not in ban_tokens]
    for _ in range(10000):
        a, b = (int(x) for x in rng.choice(pool, size=2, replace=False))
        sig = (min(a, b), max(a, b))
        if sig in used_pairs:
            continue
        used_pairs.add(sig)
        return a, b
    raise SystemExit("cannot draw unique random ABA; lower --group-size or relax uniqueness")


def assign_group_random_keys(
    clips: list[tuple[str, str]],
    true_key: dict,
    *,
    seed: int,
    group_size: int = GROUP_SIZE,
) -> tuple[dict[str, dict], dict[int, dict]]:
    """按列表顺序每 group_size 条共用一个随机 ABA。

    返回 (stem -> key_meta, group_id -> key_meta)。
    """
    gsz = max(1, int(group_size))
    rng = np.random.default_rng(int(seed))
    ban = {int(true_key["a"]), int(true_key["b"])}
    used_pairs: set[tuple[int, int]] = set()
    groups: dict[int, dict] = {}
    stem_map: dict[str, dict] = {}
    for i, (stem, _path) in enumerate(clips):
        gid = int(i // gsz)
        if gid not in groups:
            a, b = _draw_random_aba(rng, ban_tokens=ban, used_pairs=used_pairs)
            groups[gid] = {
                "group_id": gid,
                "aba": [a, b, a],
                "clusters": [-1, -1, -1],
                "a": a,
                "b": b,
                "source": f"random group={gid} seed={seed}",
                "seed": int(seed),
                "group_size": gsz,
                "clip_index_lo": int(gid * gsz),
                "clip_index_hi": int(min((gid + 1) * gsz, len(clips)) - 1),
            }
        stem_map[str(stem)] = groups[gid]
    return stem_map, groups


def l9_only_table(key: dict, *, note: str = "") -> dict:
    return {
        "note": note or key.get("source") or "L9 ABA",
        "n_layers": LAYER + 1,
        "layers": {
            str(LAYER): {
                "aba": [int(x) for x in key["aba"]],
                "clusters": [int(x) for x in (key.get("clusters") or [-1, -1, -1])],
                "token_freq": list(key.get("token_freq") or []),
                "cluster_freq": list(key.get("cluster_freq") or []),
            }
        },
    }


def clip_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if d.get("error") or d.get("metric_version") != METRIC_VERSION:
        return False
    if d.get("true_wm_aba") is None or d.get("rand_wm_aba") is None:
        return False
    return d.get("random_aba") is not None and d.get("random_group_id") is not None


def _score_true(model, wav, sr, true_table) -> dict:
    cfg = profile_score_cfg(true_table, P1_BAND_L9)
    return score_wav_once(model, wav, sr, cfg, n_peel=LAYER, period=3)


def _worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    device = str(payload["device"])
    pin_npu(device)
    log_fp, log = open_worker_log(payload)
    per_dir = Path(payload["per_clip_dir"])
    per_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(payload.get("resume"))
    true_table = payload["true_table"]
    stem_rand = dict(payload.get("stem_rand_keys") or {})
    split = str(payload.get("split") or "")
    n_ok = n_skip = n_err = 0

    try:
        set_seed(int(payload.get("seed") or 0) + int(payload["worker_id"]))
        model, sr = load_model(str(payload["moss_path"]), device)
        log(f"[{split}] start clips={len(payload['items'])} n_rand_map={len(stem_rand)}")

        for i, (stem, path) in enumerate(payload["items"]):
            safe = str(stem).replace("/", "_").replace("\\", "_")
            out_json = per_dir / f"{safe}.json"
            if resume and clip_ok(out_json):
                n_skip += 1
                n_ok += 1
                continue
            try:
                rkey = stem_rand.get(str(stem))
                if not rkey:
                    raise RuntimeError(f"missing random key assignment for stem={stem}")
                rand_table = l9_only_table(rkey, note=str(rkey.get("source") or ""))

                wav = np.asarray(read_wav(path, int(sr)), dtype=np.float32).reshape(-1)
                emb_t = embed_wav(
                    model, wav, true_table, high_start=LAYER, high_end=LAYER, layers=[LAYER], period=3
                )
                emb_r = embed_wav(
                    model, wav, rand_table, high_start=LAYER, high_end=LAYER, layers=[LAYER], period=3
                )
                wm_t = np.asarray(emb_t["wav"], dtype=np.float32).reshape(-1)
                wm_r = np.asarray(emb_r["wav"], dtype=np.float32).reshape(-1)

                so = _score_true(model, wav, int(sr), true_table)
                st = _score_true(model, wm_t, int(sr), true_table)
                srnd = _score_true(model, wm_r, int(sr), true_table)

                _write_json(
                    out_json,
                    {
                        "stem": stem,
                        "src": path,
                        "split": split,
                        "layer": LAYER,
                        "metric_version": METRIC_VERSION,
                        "sr": int(sr),
                        "random_group_id": int(rkey["group_id"]),
                        "random_aba": [int(x) for x in rkey["aba"]],
                        "n_changed_true": int(emb_t.get("n_changed") or 0),
                        "n_changed_rand": int(emb_r.get("n_changed") or 0),
                        "orig_aba": float(so["aba_embed"]),
                        "orig_slide": float(so["slide18"]),
                        "true_wm_aba": float(st["aba_embed"]),
                        "true_wm_slide": float(st["slide18"]),
                        "rand_wm_aba": float(srnd["aba_embed"]),
                        "rand_wm_slide": float(srnd["slide18"]),
                    },
                )
                n_ok += 1
            except Exception as exc:
                n_err += 1
                _write_json(
                    out_json,
                    {
                        "stem": stem,
                        "src": path,
                        "split": split,
                        "error": str(exc),
                        "trace": traceback.format_exc(),
                    },
                )
                log(f"ERR {stem}: {exc}")
            if (i + 1) % 20 == 0 or i == 0:
                log(f"[{split}] {i+1}/{len(payload['items'])} ok={n_ok} skip={n_skip} err={n_err}")
    finally:
        free_device()
        if log_fp:
            log_fp.close()
    return {"worker_id": payload["worker_id"], "n_ok": n_ok, "n_skip": n_skip, "n_err": n_err}


def load_rows(per_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(per_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("error") or d.get("metric_version") != METRIC_VERSION:
            continue
        if d.get("true_wm_aba") is None or d.get("rand_wm_aba") is None:
            continue
        rows.append(d)
    return rows


def _arr(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows if r.get(key) is not None], dtype=np.float64)


def calibrate(rows: list[dict], *, target_far: float = TARGET_FAR) -> dict:
    pos_a, neg_a = _arr(rows, "true_wm_aba"), _arr(rows, "rand_wm_aba")
    pos_s, neg_s = _arr(rows, "true_wm_slide"), _arr(rows, "rand_wm_slide")
    ha = roc_threshold_at_far(pos_a, neg_a, target_far)
    hs = roc_threshold_at_far(pos_s, neg_s, target_far)
    return {
        "target_far": float(target_far),
        "n_pos": int(pos_a.size),
        "n_neg": int(neg_a.size),
        "pos": "true_wm scored by true_key",
        "neg": "random_wm scored by true_key",
        "aba": {
            "T": ha.get("T"),
            "calib_far": ha.get("far"),
            "calib_tpr": ha.get("tpr"),
            "pos_mean": float(np.mean(pos_a)) if pos_a.size else None,
            "neg_mean": float(np.mean(neg_a)) if neg_a.size else None,
        },
        "slide": {
            "T": hs.get("T"),
            "calib_far": hs.get("far"),
            "calib_tpr": hs.get("tpr"),
            "pos_mean": float(np.mean(pos_s)) if pos_s.size else None,
            "neg_mean": float(np.mean(neg_s)) if neg_s.size else None,
        },
    }


def evaluate(rows: list[dict], calib: dict) -> dict:
    pos_a, neg_a = _arr(rows, "true_wm_aba"), _arr(rows, "rand_wm_aba")
    pos_s, neg_s = _arr(rows, "true_wm_slide"), _arr(rows, "rand_wm_slide")
    t_a = (calib.get("aba") or {}).get("T")
    t_s = (calib.get("slide") or {}).get("T")
    out: dict = {"n_pos": int(pos_a.size), "n_neg": int(neg_a.size), "T_aba": t_a, "T_slide": t_s}
    if t_a is not None:
        out["aba"] = {
            "T": float(t_a),
            "far": rate_pack(int(np.sum(neg_a >= t_a)), int(neg_a.size)),
            "tpr": rate_pack(int(np.sum(pos_a >= t_a)), int(pos_a.size)),
            "auc": float(auc_mw(pos_a, neg_a)) if pos_a.size and neg_a.size else None,
            "pos_mean": float(np.mean(pos_a)) if pos_a.size else None,
            "neg_mean": float(np.mean(neg_a)) if neg_a.size else None,
        }
    if t_s is not None:
        out["slide"] = {
            "T": float(t_s),
            "far": rate_pack(int(np.sum(neg_s >= t_s)), int(neg_s.size)),
            "tpr": rate_pack(int(np.sum(pos_s >= t_s)), int(pos_s.size)),
            "auc": float(auc_mw(pos_s, neg_s)) if pos_s.size and neg_s.size else None,
            "pos_mean": float(np.mean(pos_s)) if pos_s.size else None,
            "neg_mean": float(np.mean(neg_s)) if neg_s.size else None,
        }
    return out


def _fmt_pct(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, dict):
        v = x.get("value")
        return "—" if v is None else f"{100 * float(v):.2f}%"
    return f"{100 * float(x):.2f}%"


def _fmt_num(x, nd: int = 4) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def render_md(keys: dict, calib: dict, eval_m: dict | None, paths: dict) -> str:
    ta = (calib or {}).get("aba") or {}
    ts = (calib or {}).get("slide") or {}
    ea = ((eval_m or {}) or {}).get("aba") or {}
    es = ((eval_m or {}) or {}).get("slide") or {}
    tk = keys.get("true") or {}
    sch = keys.get("schedule") or {}
    return "\n".join(
        [
            "# L9 真 key vs 随机水印 · FAR01",
            "",
            f"- wav_1500: `{paths.get('wav_1500')}`",
            f"- wav_2040: `{paths.get('wav_2040')}`",
            f"- out: `{paths.get('out_dir')}`",
            f"- true ABA: `{tk.get('aba')}`",
            f"- random: 每 {sch.get('group_size', GROUP_SIZE)} 条一个随机 ABA"
            f"（calib 组数={sch.get('n_groups_calib')}，eval 组数={sch.get('n_groups_eval')}）",
            f"- 正类 true_wm / 负类 random_wm；检测一律用 true_key；target FAR={TARGET_FAR:.0%}",
            f"- per-clip: `calib/per_clip/*.json` · `eval/per_clip/*.json`（含 random_group_id / random_aba）",
            "",
            "## 定标（1500）",
            "",
            "| 通道 | T | calib FAR | calib TPR | pos_mean | neg_mean | n_pos/n_neg |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| aba_embed | {_fmt_num(ta.get('T'))} | {_fmt_pct(ta.get('calib_far'))} | "
            f"{_fmt_pct(ta.get('calib_tpr'))} | {_fmt_num(ta.get('pos_mean'))} | "
            f"{_fmt_num(ta.get('neg_mean'))} | {calib.get('n_pos')}/{calib.get('n_neg')} |",
            f"| slide_v3 | {_fmt_num(ts.get('T'))} | {_fmt_pct(ts.get('calib_far'))} | "
            f"{_fmt_pct(ts.get('calib_tpr'))} | {_fmt_num(ts.get('pos_mean'))} | "
            f"{_fmt_num(ts.get('neg_mean'))} | |",
            "",
            "## 评测（2040）",
            "",
            "| 通道 | T(冻结) | FAR↓ | TPR↑ | AUC↑ | pos_mean | neg_mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| aba_embed | {_fmt_num(ea.get('T'))} | {_fmt_pct(ea.get('far'))} | "
            f"{_fmt_pct(ea.get('tpr'))} | {_fmt_num(ea.get('auc'))} | "
            f"{_fmt_num(ea.get('pos_mean'))} | {_fmt_num(ea.get('neg_mean'))} |",
            f"| slide_v3 | {_fmt_num(es.get('T'))} | {_fmt_pct(es.get('far'))} | "
            f"{_fmt_pct(es.get('tpr'))} | {_fmt_num(es.get('auc'))} | "
            f"{_fmt_num(es.get('pos_mean'))} | {_fmt_num(es.get('neg_mean'))} |",
            "",
        ]
    )


def run_split(
    *,
    split: str,
    wav_dir: Path,
    out_split: Path,
    true_table: dict,
    true_key: dict,
    limit: int,
    args,
) -> tuple[list[dict], dict[int, dict]]:
    if not wav_dir.is_dir():
        raise SystemExit(f"[{split}] wav dir not found: {wav_dir}")
    out_split.mkdir(parents=True, exist_ok=True)
    per_dir = out_split / "per_clip"
    per_dir.mkdir(parents=True, exist_ok=True)
    clips = list_clips(wav_dir, limit=int(limit))
    if not clips:
        raise SystemExit(f"[{split}] no clips in {wav_dir}")

    # 不同 split 用不同子种子，避免 1500/2040 组 key 完全相同；组内仍多样
    split_seed = int(args.seed) + (0 if split == "calib" else 10007)
    stem_map, groups = assign_group_random_keys(
        clips, true_key, seed=split_seed, group_size=int(args.group_size)
    )
    dump_json(
        out_split / "random_key_schedule.json",
        {
            "split": split,
            "seed": split_seed,
            "group_size": int(args.group_size),
            "n_clips": len(clips),
            "n_groups": len(groups),
            "groups": {str(k): v for k, v in sorted(groups.items())},
            "stem_to_group": {s: int(m["group_id"]) for s, m in stem_map.items()},
        },
    )
    print(
        f"[{split}] n={len(clips)} groups={len(groups)} "
        f"group_size={args.group_size} -> {per_dir}",
        flush=True,
    )

    n_workers = max(1, int(args.num_workers))
    logs = out_split / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # 分片时只带本 shard 需要的 stem→key，减小 payload
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
            "true_table": true_table,
            "split": split,
        },
    )
    for pl in payloads:
        need = {str(stem) for stem, _ in pl.get("items") or []}
        pl["stem_rand_keys"] = {s: stem_map[s] for s in need if s in stem_map}

    outs = run_shards(_worker, payloads, num_workers=n_workers)
    dump_json(out_split / "workers.json", outs)
    rows = load_rows(per_dir)
    dump_json(
        out_split / "n_rows.json",
        {"n": len(rows), "wav_dir": str(wav_dir), "n_groups": len(groups)},
    )
    return rows, groups


def parse_args():
    p = argparse.ArgumentParser(
        description="L9 true vs random FAR01：--wav-1500 / --wav-2040 / --out-dir"
    )
    p.add_argument("--wav-1500", type=str, required=True, help="定标集目录（mix_1500）")
    p.add_argument("--wav-2040", type=str, required=True, help="评测集目录（fleurs_2040）")
    p.add_argument("--out-dir", type=str, required=True, help="结果输出根目录")
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--table", type=str, default=default_table_path())
    p.add_argument("--seed", type=int, default=0, help="随机 ABA 总种子（各组由此派生）")
    p.add_argument(
        "--group-size",
        type=int,
        default=GROUP_SIZE,
        help="每多少条语音共用一个随机 ABA（默认 20）",
    )
    p.add_argument("--target-far", type=float, default=TARGET_FAR)
    p.add_argument("--calib-limit", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--stage", type=str, default="all", choices=("all", "calib", "eval", "summary"))
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    add_npu_args(p, default_workers=8, default_device="npu:0")
    return p.parse_args()


def main() -> None:
    ensure_spawn()
    args = parse_args()
    if args.no_resume:
        args.resume = False

    wav_1500 = Path(args.wav_1500)
    wav_2040 = Path(args.wav_2040)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = out_dir / "calib"
    eval_dir = out_dir / "eval"

    full_table = ensure_rare_table(str(args.table))
    true_key = true_l9_from_table(full_table)
    true_table = full_table

    keys_doc: dict = {
        "layer": LAYER,
        "true": true_key,
        "table_path": str(args.table),
        "schedule": {
            "group_size": int(args.group_size),
            "seed": int(args.seed),
            "rule": "every group_size clips share one random ABA; calib/eval use different sub-seeds",
        },
        "paths": {
            "wav_1500": str(wav_1500),
            "wav_2040": str(wav_2040),
            "out_dir": str(out_dir),
        },
        "note": "detect with true_key; pos=true_wm; neg=random_wm (per-group random ABA)",
    }
    dump_json(out_dir / "keys.json", keys_doc)
    print(
        f"true ABA={true_key['aba']}  random=every {args.group_size} clips a new ABA",
        flush=True,
    )
    print(f"1500={wav_1500}\n2040={wav_2040}\nout={out_dir}", flush=True)

    stage = str(args.stage)
    calib = None
    eval_m = None
    n_g_calib = n_g_eval = None

    if stage in ("all", "calib"):
        rows_c, groups_c = run_split(
            split="calib",
            wav_dir=wav_1500,
            out_split=calib_dir,
            true_table=true_table,
            true_key=true_key,
            limit=int(args.calib_limit),
            args=args,
        )
        n_g_calib = len(groups_c)
        calib = calibrate(rows_c, target_far=float(args.target_far))
        dump_json(calib_dir / "threshold.json", calib)
        dump_json(out_dir / "threshold.json", calib)
        print(
            f"[calib] T_aba={calib['aba'].get('T')} "
            f"far={calib['aba'].get('calib_far')} tpr={calib['aba'].get('calib_tpr')} "
            f"n_groups={n_g_calib} per_clip={calib_dir / 'per_clip'}",
            flush=True,
        )
    else:
        th = out_dir / "threshold.json"
        if not th.is_file():
            th = calib_dir / "threshold.json"
        if not th.is_file():
            raise SystemExit("missing threshold.json；先跑 --stage calib")
        calib = json.loads(th.read_text(encoding="utf-8"))

    if stage in ("all", "eval"):
        rows_e, groups_e = run_split(
            split="eval",
            wav_dir=wav_2040,
            out_split=eval_dir,
            true_table=true_table,
            true_key=true_key,
            limit=int(args.eval_limit),
            args=args,
        )
        n_g_eval = len(groups_e)
        eval_m = evaluate(rows_e, calib)
        eval_m["wav_2040"] = str(wav_2040)
        eval_m["n_random_groups"] = n_g_eval
        dump_json(eval_dir / "metrics.json", eval_m)
        dump_json(out_dir / "eval_metrics.json", eval_m)
        ea = eval_m.get("aba") or {}
        print(
            f"[eval] FAR={ea.get('far')} TPR={ea.get('tpr')} AUC={ea.get('auc')} "
            f"n_groups={n_g_eval} per_clip={eval_dir / 'per_clip'}",
            flush=True,
        )
    elif stage == "summary":
        rows_e = load_rows(eval_dir / "per_clip")
        if rows_e:
            eval_m = evaluate(rows_e, calib)
            dump_json(eval_dir / "metrics.json", eval_m)
            dump_json(out_dir / "eval_metrics.json", eval_m)

    keys_doc["schedule"]["n_groups_calib"] = n_g_calib
    keys_doc["schedule"]["n_groups_eval"] = n_g_eval
    dump_json(out_dir / "keys.json", keys_doc)

    paths = {"wav_1500": str(wav_1500), "wav_2040": str(wav_2040), "out_dir": str(out_dir)}
    (out_dir / "FINDINGS.md").write_text(
        render_md(keys_doc, calib or {}, eval_m, paths), encoding="utf-8"
    )
    (out_dir / "RUN.md").write_text(
        "# L9 true vs random FAR01\n\n"
        f"- `--wav-1500`: `{wav_1500}`\n"
        f"- `--wav-2040`: `{wav_2040}`\n"
        f"- `--out-dir`: `{out_dir}`\n"
        f"- random: every `{args.group_size}` clips → one ABA "
        f"(calib groups={n_g_calib}, eval groups={n_g_eval})\n"
        f"- per-clip: `calib/per_clip/` · `eval/per_clip/`\n"
        f"- schedule: `calib/random_key_schedule.json` · `eval/random_key_schedule.json`\n",
        encoding="utf-8",
    )
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
