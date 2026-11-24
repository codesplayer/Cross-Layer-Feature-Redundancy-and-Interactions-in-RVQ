"""汇总 MOSS / DAC / Qwen 干净重构相对原波形的 LSD、MCD。

读 ``rvq_extra_eta_n1500/{codec}/{stem}.json`` 里已有的 ``lsd`` / ``mcd``
（encode→decode 对原语音）。不再编解码。

写出：
- ``{out}/summary.json``
- ``{out}/FINDINGS.md``
- ``{out}/per_clip/{stem}.json``（同一句三个 codec 并排）

运行::

  $env:PYTHONPATH="e:\\code\\water_marks"
  python -m other.experiments.summarize_codecs_recon_lsd_mcd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from base.config import EXP_ROOT
from base.util import dump_json

CODECS = ("moss", "dac", "qwen")
_IN_CANDIDATES = (
    Path(r"E:\code\water_marks\result\rvq\rvq_extra_eta_n1500"),
    Path(r"/data1/home/ynian/codes/eval/add_exp/result/rvq_extra_eta_n1500"),
    EXP_ROOT / "rvq_extra_eta_n1500",
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / "rvq" / "rvq_extra_eta_n1500",
)
OUT_STEM = "codecs_recon_lsd_mcd_n1500"
_OUT_CANDIDATES = (
    Path(r"E:\code\water_marks\result") / OUT_STEM,
    EXP_ROOT / OUT_STEM,
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
)


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{float(x):.4f}"


def first_existing(*paths: Path) -> Path:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return Path(paths[0])


def load_codec_rows(root: Path, codec: str) -> dict[str, dict]:
    folder = Path(root) / codec
    out: dict[str, dict] = {}
    if not folder.is_dir():
        return out
    for fp in folder.glob("*.json"):
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict) or doc.get("error"):
            continue
        if doc.get("lsd") is None or doc.get("mcd") is None:
            continue
        stem = str(doc.get("stem") or doc.get("name") or fp.stem)
        out[stem] = {
            "stem": stem,
            "codec": codec,
            "lsd": float(doc["lsd"]),
            "mcd": float(doc["mcd"]),
        }
    return out


def desc(vals) -> dict:
    v = np.asarray([float(x) for x in vals if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p10": float(np.percentile(v, 10)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
    }


def aggregate(by_codec: dict[str, dict[str, dict]]) -> dict:
    codecs = {}
    for codec in CODECS:
        rows = list(by_codec.get(codec, {}).values())
        codecs[codec] = {
            "n": len(rows),
            "lsd": desc([r["lsd"] for r in rows]),
            "mcd": desc([r["mcd"] for r in rows]),
        }
    stems = set(by_codec.get(CODECS[0], {}))
    for codec in CODECS[1:]:
        stems &= set(by_codec.get(codec, {}))
    paired = {"n": len(stems)}
    for codec in CODECS:
        paired[codec] = {
            "lsd": desc([by_codec[codec][s]["lsd"] for s in stems]),
            "mcd": desc([by_codec[codec][s]["mcd"] for s in stems]),
        }
    return {
        "source": "rvq_extra_eta_n1500 encode-decode vs original wav",
        "codecs": codecs,
        "paired_stems": paired,
    }


def render(summary: dict) -> str:
    lines = [
        "# 三 codec 重构 LSD / MCD（n1500）\n\n",
        "干净 encode→decode 相对原波形。数字来自 `rvq_extra_eta_n1500` 每句 JSON，没有重跑编解码。\n\n",
        "| codec | n | LSD mean | LSD p50 | LSD p10–p90 | MCD mean | MCD p50 | MCD p10–p90 |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for codec in CODECS:
        e = (summary.get("codecs") or {}).get(codec) or {}
        lsd, mcd = e.get("lsd") or {}, e.get("mcd") or {}
        lines.append(
            f"| {codec} | {e.get('n', 0)} | {_fmt(lsd.get('mean'))} | {_fmt(lsd.get('p50'))} | "
            f"{_fmt(lsd.get('p10'))}–{_fmt(lsd.get('p90'))} | {_fmt(mcd.get('mean'))} | "
            f"{_fmt(mcd.get('p50'))} | {_fmt(mcd.get('p10'))}–{_fmt(mcd.get('p90'))} |\n"
        )
    paired = summary.get("paired_stems") or {}
    lines.append(f"\n三套都有的句子：n={paired.get('n', 0)}。\n")
    return "".join(lines)


def write_per_clip(out_dir: Path, by_codec: dict[str, dict[str, dict]]) -> int:
    per_dir = out_dir / "per_clip"
    per_dir.mkdir(parents=True, exist_ok=True)
    stems = set()
    for rows in by_codec.values():
        stems.update(rows)
    n = 0
    for stem in sorted(stems):
        doc = {"name": stem, "recon": {}}
        for codec in CODECS:
            rec = (by_codec.get(codec) or {}).get(stem)
            if rec:
                doc["recon"][codec] = {"lsd": rec["lsd"], "mcd": rec["mcd"]}
        dump_json(per_dir / f"{stem}.json", doc)
        n += 1
    return n


def parse_args():
    p = argparse.ArgumentParser(description="Summarize moss/dac/qwen recon LSD and MCD")
    p.add_argument("--root", type=str, default="", help="rvq_extra_eta_n1500 根目录")
    p.add_argument("--out_dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root) if args.root else first_existing(*_IN_CANDIDATES)
    if not root.is_dir():
        raise SystemExit(f"missing input dir: {root}")
    out_dir = Path(args.out_dir) if args.out_dir else first_existing(*_OUT_CANDIDATES)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif not out_dir.parent.exists():
        out_dir = _OUT_CANDIDATES[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    by_codec = {codec: load_codec_rows(root, codec) for codec in CODECS}
    for codec, rows in by_codec.items():
        print(f"{codec} clips={len(rows)} from {root / codec}", flush=True)
    summary = aggregate(by_codec)
    summary["input"] = str(root)
    dump_json(out_dir / "summary.json", summary)
    (out_dir / "FINDINGS.md").write_text(render(summary), encoding="utf-8")
    n_clip = write_per_clip(out_dir, by_codec)
    print(f"per_clip={n_clip} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
