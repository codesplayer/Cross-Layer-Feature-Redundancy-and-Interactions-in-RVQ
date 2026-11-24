#!/usr/bin/env python3
"""paper_P1_RVQ 统一入口。

用法::

  # 小规模冒烟（默认）：路径检查 + 几何恒等式 + ABA 嵌入/检测
  python main.py
  python main.py smoke --n-clips 1 --device cpu

  # 列出可调度的规模实验（需自行加参数）
  python main.py list

  # 转发到子实验模块
  python main.py geometry -- --codecs moss --n-clips 3 --num-workers 1 --device cuda:0
  python main.py layer-gap -- --codecs moss --n-clips 3
  python main.py readback -- --codecs moss --n-clips 3
  python main.py watermark-attack -- --help

说明：本入口只保证 ``smoke`` 在本包 PYTHONPATH 下自洽；
规模实验脚本可能还依赖本机数据路径 / 多卡环境（见 README）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Optional local deps (when conda env is not writable): paper_P1_RVQ/.deps
_DEPS = ROOT / ".deps"
if _DEPS.is_dir() and str(_DEPS) not in sys.path:
    sys.path.insert(0, str(_DEPS))


def _resolve_wav(wav_dir: str | None, name: str) -> Path:
    from base.config import NAMES, first_existing

    candidates = []
    if wav_dir:
        candidates.append(Path(wav_dir) / f"{name}.wav")
    candidates.extend(
        [
            Path(r"e:\data\Clotho\aishell1\test\wav") / f"{name}.wav",
            Path(r"d:\data\codec_an\exp_watermark\data\mix_1500") / f"{name}.wav",
            Path(r"/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500") / f"{name}.wav",
        ]
    )
    for stem in NAMES:
        candidates.append(Path(r"e:\data\Clotho\aishell1\test\wav") / f"{stem}.wav")
    hit = first_existing(*candidates)
    p = Path(hit)
    if not p.exists():
        raise FileNotFoundError(f"wav not found for {name!r}; tried {candidates[:4]} ...")
    return p


def _pick_device(explicit: str | None) -> str:
    from base.util import auto_device

    return (explicit or "").strip() or auto_device()


def cmd_list(_: argparse.Namespace) -> int:
    rows = [
        ("smoke", "路径检查 + 几何恒等式 + ABA embed/detect（本入口实现）"),
        ("geometry", "other.experiments.run_rvq_af_n1500  （能量/层对cosine/C⊥）"),
        ("layer-gap", "P1.experiments.run_codecs_band_layer_gap  （ΔA / 编辑代价）"),
        ("two-layer", "P1.experiments.run_codecs_band_two_layer_random"),
        ("readback", "P1.experiments.run_codecs_multi_roundtrip_feat_n1500"),
        ("watermark-attack", "P1.experiments.run_anchor_attack_three_n1500"),
        ("watermark-quality", "P1.experiments.run_wm_clean_quality"),
        ("fleurs-far01", "P1.experiments.eval_fleurs_far01_from_anchor_clean"),
    ]
    print("paper_P1_RVQ entry commands:\n")
    for name, desc in rows:
        print(f"  {name:20s}  {desc}")
    print("\nForwarding example:")
    print("  python main.py geometry -- --codecs moss --n-clips 3 --num-workers 1 --device cpu")
    return 0


def _run_module(module: str, forward: list[str]) -> int:
    import runpy

    sys.argv = [module] + list(forward)
    print(f"[main] exec module {module} argv={sys.argv}")
    runpy.run_module(module, run_name="__main__")
    return 0


FORWARD = {
    "geometry": "other.experiments.run_rvq_af_n1500",
    "layer-gap": "P1.experiments.run_codecs_band_layer_gap",
    "two-layer": "P1.experiments.run_codecs_band_two_layer_random",
    "readback": "P1.experiments.run_codecs_multi_roundtrip_feat_n1500",
    "watermark-attack": "P1.experiments.run_anchor_attack_three_n1500",
    "watermark-quality": "P1.experiments.run_wm_clean_quality",
    "fleurs-far01": "P1.experiments.eval_fleurs_far01_from_anchor_clean",
}


def cmd_smoke(args: argparse.Namespace) -> int:
    from base.config import NAMES, default_model_path, default_table_path
    from base.moss_io import read_wav
    from base.residual_feat import encode_hidden
    from base.util import load_model
    from other.experiments.rvq_walk import walk_rvq
    from base.moss_io import encode_moss, decode_moss
    from base.residual_feat import band_zq_of, codes_to_zq_sum, feat_cosine
    from P1.embed import embed_aba

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "result_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"ok": True, "steps": [], "device": None, "paths": {}}

    def step(name: str, ok: bool, detail: dict | None = None) -> None:
        item = {"name": name, "ok": bool(ok), **(detail or {})}
        report["steps"].append(item)
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            report["ok"] = False

    # --- 0. paths ---
    model_path = Path(args.model_path or default_model_path())
    table_path = Path(args.table or default_table_path())
    names = list(NAMES)[: max(1, int(args.n_clips))]
    wavs = []
    try:
        for n in names:
            wavs.append(_resolve_wav(args.wav_dir, n))
        step(
            "resolve_paths",
            model_path.exists() and table_path.exists() and all(p.exists() for p in wavs),
            {
                "model": str(model_path),
                "table": str(table_path),
                "wavs": [str(p) for p in wavs],
            },
        )
    except Exception as e:
        step("resolve_paths", False, {"error": str(e)})
        _write_report(out_dir, report)
        return 1

    report["paths"] = {
        "model": str(model_path),
        "table": str(table_path),
        "wavs": [str(p) for p in wavs],
    }

    device = _pick_device(args.device)
    report["device"] = device
    print(f"[main] device={device}")

    t0 = time.time()
    try:
        model, sr = load_model(str(model_path), device)
        step("load_moss", True, {"sr": int(sr), "sec": round(time.time() - t0, 2)})
    except Exception as e:
        step("load_moss", False, {"error": str(e), "trace": traceback.format_exc()[-800:]})
        _write_report(out_dir, report)
        return 1

    # --- 1. geometry identity on first clip ---
    try:
        wav = read_wav(wavs[0], sr)
        hidden, lengths = encode_hidden(model, wav)
        codes, zqs, _pres = walk_rvq(model, hidden, lengths)
        # z: encoder feature in quantizer space
        import torch

        with torch.no_grad():
            z = model.quantizer.input_proj(hidden)[0].detach().float().cpu().numpy()
        T = zqs[0].shape[1]
        z = z[:, :T]
        Q = sum(zqs)
        r = z - Q
        # framewise: ||z||^2 = ||Q||^2 + ||r||^2 + 2<Q,r>
        zn = (z * z).sum(0)
        Qn = (Q * Q).sum(0)
        rn = (r * r).sum(0)
        qr = 2.0 * (Q * r).sum(0)
        err = float(((Qn + rn + qr) - zn).mean())
        # E1 vs E2 cross: sum_l ||ql||^2 vs ||Q||^2
        e1 = sum((q * q).sum(0) for q in zqs)
        e2 = Qn
        # identity E2 = E1 + E+ + E-  checked via reconstruction of cross
        cross = e2 - e1  # should equal 2 sum_{i<j} <qi,qj>
        pair = 0.0
        L = len(zqs)
        for i in range(L):
            for j in range(i + 1, L):
                pair = pair + 2.0 * (zqs[i] * zqs[j]).sum(0)
        cross_err = float((pair - cross).mean())
        ok_geom = abs(err) < 1e-3 and abs(cross_err) < 1e-3
        step(
            "geometry_identity",
            ok_geom,
            {
                "n_q": int(L),
                "T": int(T),
                "energy_id_err": err,
                "cross_id_err": cross_err,
                "token_L9_head": codes[9, :8].tolist() if codes.shape[0] > 9 else None,
            },
        )
    except Exception as e:
        step("geometry_identity", False, {"error": str(e), "trace": traceback.format_exc()[-800:]})

    # --- 2. ABA watermark embed + detect ---
    try:
        layer = int(args.layer)
        emb = embed_aba(model, wav, layer=layer, table_path=str(table_path))
        # Template from the written token grid (same T as the utterance).
        tmpl = codes_to_zq_sum(model, emb["mod"], layer, layer, layers=[layer])
        # Direct (pre-decode) and after one encode-decode round trip.
        score_direct = feat_cosine(band_zq_of(model, emb["wav"], layer, layer), tmpl)
        wm_re = decode_moss(model, encode_moss(model, emb["wav"]))
        clean_re = decode_moss(model, encode_moss(model, wav))
        score_wm = feat_cosine(band_zq_of(model, wm_re, layer, layer), tmpl)
        score_clean = feat_cosine(band_zq_of(model, clean_re, layer, layer), tmpl)
        codes_re = encode_moss(model, emb["wav"])
        token_agree = float((codes_re[layer] == emb["mod"][layer]).mean())
        s_dir = float(score_direct["cosine"])
        s_wm = float(score_wm["cosine"])
        s_cl = float(score_clean["cosine"])
        gap = s_wm - s_cl
        ok_wm = gap > float(args.min_gap) and s_dir > 0.3
        step(
            "aba_embed_detect",
            ok_wm and emb["wav"].shape[0] > 0,
            {
                "layer": layer,
                "n_changed": int(emb["n_changed"]),
                "score_direct": s_dir,
                "score_wm_reenc": s_wm,
                "score_clean_reenc": s_cl,
                "gap": gap,
                "token_agree_L": token_agree,
                "min_gap": float(args.min_gap),
            },
        )
        if args.save_wav:
            from base.io import write_wav_float

            write_wav_float(out_dir / f"smoke_wm_L{layer}.wav", emb["wav"], sr)
    except Exception as e:
        step("aba_embed_detect", False, {"error": str(e), "trace": traceback.format_exc()[-800:]})

    report["elapsed_sec"] = round(time.time() - t0, 2)
    path = _write_report(out_dir, report)
    print(f"[main] report -> {path}")
    print("[main] SMOKE " + ("OK" if report["ok"] else "FAILED"))
    return 0 if report["ok"] else 1


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _write_report(out_dir: Path, report: dict) -> Path:
    path = out_dir / "smoke_report.json"
    path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="paper_P1_RVQ unified entry")
    sub = p.add_subparsers(dest="cmd")

    p_smoke = sub.add_parser("smoke", help="small-scale verification (default)")
    p_smoke.add_argument("--device", type=str, default="")
    p_smoke.add_argument("--model_path", type=str, default="")
    p_smoke.add_argument("--table", type=str, default="")
    p_smoke.add_argument("--wav-dir", type=str, default=r"e:\data\Clotho\aishell1\test\wav")
    p_smoke.add_argument("--n-clips", type=int, default=1)
    p_smoke.add_argument("--layer", type=int, default=9)
    p_smoke.add_argument("--min-gap", type=float, default=0.05, help="min (score_wm - score_clean)")
    p_smoke.add_argument("--out-dir", type=str, default="")
    p_smoke.add_argument("--save-wav", action="store_true")

    sub.add_parser("list", help="list entry commands")

    for name in FORWARD:
        sp = sub.add_parser(name, help=f"forward to {FORWARD[name]}")
        sp.add_argument("forward", nargs=argparse.REMAINDER, help="args after --")

    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # default command = smoke
    if not argv or argv[0].startswith("-"):
        argv = ["smoke"] + argv
    args = build_parser().parse_args(argv)

    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "smoke":
        return cmd_smoke(args)
    if args.cmd in FORWARD:
        forward = list(getattr(args, "forward", []) or [])
        if forward and forward[0] == "--":
            forward = forward[1:]
        return _run_module(FORWARD[args.cmd], forward)

    build_parser().print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
