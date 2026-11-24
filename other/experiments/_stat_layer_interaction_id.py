"""Compute I_D = D_ij - D_i - D_j + D_clean for LSD/MCD.

D_i, D_j  : codecs_band_layer_gap (single-layer ABA)
D_ij      : codecs_band_two_layer_random
D_clean   : codecs_recon_lsd_mcd_n1500 (unmodified codec recon)
            lsd_mcd/raw is moss-only cross-check
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(r"D:\KimiData\kimi\Workspaces\P1\exp")
SINGLE = ROOT / "codecs_band_layer_gap" / "summary.json"
PAIR = ROOT / "codecs_band_two_layer_random" / "summary.json"
CLEAN_DIR = ROOT / "codecs_recon_lsd_mcd_n1500" / "per_clip"
LSD_MCD = ROOT / "lsd_mcd" / "summary.json"
OUT_DIR = ROOT / "layer_interaction_id"
CODECS = ("moss", "qwen", "dac")


def mean(xs: list[float]) -> float | None:
    return float(statistics.mean(xs)) if xs else None


def load_clean() -> dict[str, dict[str, float]]:
    """codec -> {lsd, mcd, n} from per-clip recon."""
    acc: dict[str, dict[str, list[float]]] = {c: {"lsd": [], "mcd": []} for c in CODECS}
    for path in sorted(CLEAN_DIR.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        recon = rec.get("recon") or {}
        for codec in CODECS:
            row = recon.get(codec) or {}
            if row.get("lsd") is not None:
                acc[codec]["lsd"].append(float(row["lsd"]))
            if row.get("mcd") is not None:
                acc[codec]["mcd"].append(float(row["mcd"]))
    out = {}
    for codec in CODECS:
        out[codec] = {
            "lsd": mean(acc[codec]["lsd"]),
            "mcd": mean(acc[codec]["mcd"]),
            "n": len(acc[codec]["lsd"]),
            "source": str(CLEAN_DIR),
        }
    return out


def load_single(summary: dict) -> dict[str, dict[int, dict[str, float]]]:
    out: dict[str, dict[int, dict[str, float]]] = {}
    for codec in CODECS:
        by = {}
        for row in summary["systems"][codec]["by_layer"]:
            layer = int(row["write_layer"])
            by[layer] = {
                "lsd": float(row["mean_lsd"]),
                "mcd": float(row["mean_mcd"]),
                "n": int(row["n"]),
            }
        out[codec] = by
    return out


def parse_pair_key(key: str) -> tuple[int, int]:
    # L0_L12 or L04_L05
    a, b = key.split("_", 1)
    return int(a.lstrip("Ll")), int(b.lstrip("Ll"))


def load_pairs(summary: dict) -> dict[str, dict[tuple[int, int], dict[str, float]]]:
    out: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    for codec in CODECS:
        by = {}
        for key, row in summary["systems"][codec]["aggregate"]["by_pair"].items():
            i, j = parse_pair_key(key)
            by[(i, j)] = {
                "lsd": float(row["mean_lsd"]),
                "mcd": float(row["mean_mcd"]),
                "n": int(row["n"]),
                "key": key,
            }
        out[codec] = by
    return out


def summarize_id(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    arr = sorted(vals)
    return {
        "n": len(arr),
        "mean": float(statistics.mean(arr)),
        "std": float(statistics.pstdev(arr)) if len(arr) > 1 else 0.0,
        "min": float(arr[0]),
        "p10": float(arr[max(0, int(0.1 * (len(arr) - 1)))]),
        "p50": float(arr[len(arr) // 2]),
        "p90": float(arr[min(len(arr) - 1, int(0.9 * (len(arr) - 1)))]),
        "max": float(arr[-1]),
        "frac_pos": float(sum(1 for x in arr if x > 0) / len(arr)),
        "frac_neg": float(sum(1 for x in arr if x < 0) / len(arr)),
        "frac_near0": float(sum(1 for x in arr if abs(x) < 0.05) / len(arr)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = load_clean()
    single = load_single(json.loads(SINGLE.read_text(encoding="utf-8")))
    pairs = load_pairs(json.loads(PAIR.read_text(encoding="utf-8")))

    moss_lsd_mcd = None
    if LSD_MCD.exists():
        raw = json.loads(LSD_MCD.read_text(encoding="utf-8"))["systems"]["raw"]
        moss_lsd_mcd = {"lsd": float(raw["mean_lsd"]), "mcd": float(raw["mean_mcd"]), "n": int(raw["n_ok"])}

    result: dict = {
        "formula": "I_D = D_ij - D_i - D_j + D_clean",
        "readout": "I_D>0 超加性（两层比两层之和更伤）；I_D<0 亚加性（两层互相抵消一部分）",
        "D_clean": clean,
        "D_clean_lsd_mcd_moss_crosscheck": moss_lsd_mcd,
        "sources": {
            "single": str(SINGLE),
            "pair": str(PAIR),
            "clean": str(CLEAN_DIR),
        },
        "codecs": {},
    }

    md = [
        "# 单层 × 双层失真交互 \(I_D\)",
        "",
        r"\[ I_D = D_{ij} - D_i - D_j + D_{\mathrm{clean}} \]",
        "",
        "其中 \(D\) 为相对原始语音的失真（LSD 或 MCD）。",
        "- \(D_i, D_j\)：`codecs_band_layer_gap` 单层 ABA 均值",
        "- \(D_{ij}\)：`codecs_band_two_layer_random` 双层 ABA 均值",
        "- \(D_{\\mathrm{clean}}\)：`codecs_recon_lsd_mcd_n1500` 未改 codec 重构均值",
        "",
        "均值上 \(I_D\) 与「先逐句算再平均」相同（每条件都是 1500 句）。",
        "",
        "## \(D_{\\mathrm{clean}}\)",
        "",
        "| codec | LSD | MCD | n |",
        "|---|---:|---:|---:|",
    ]
    for codec in CODECS:
        c = clean[codec]
        md.append(f"| {codec} | {c['lsd']:.4f} | {c['mcd']:.4f} | {c['n']} |")
    if moss_lsd_mcd:
        md += [
            "",
            f"交叉核对：`lsd_mcd` 里 moss raw LSD={moss_lsd_mcd['lsd']:.4f}, MCD={moss_lsd_mcd['mcd']:.4f}。",
        ]

    for codec in CODECS:
        dc = clean[codec]
        rows = []
        for (i, j), dij in sorted(pairs[codec].items()):
            if i not in single[codec] or j not in single[codec]:
                continue
            di = single[codec][i]
            dj = single[codec][j]
            row = {
                "pair": f"L{i}_L{j}",
                "i": i,
                "j": j,
                "n": dij["n"],
                "D_ij_lsd": dij["lsd"],
                "D_i_lsd": di["lsd"],
                "D_j_lsd": dj["lsd"],
                "D_clean_lsd": dc["lsd"],
                "I_lsd": dij["lsd"] - di["lsd"] - dj["lsd"] + dc["lsd"],
                "D_ij_mcd": dij["mcd"],
                "D_i_mcd": di["mcd"],
                "D_j_mcd": dj["mcd"],
                "D_clean_mcd": dc["mcd"],
                "I_mcd": dij["mcd"] - di["mcd"] - dj["mcd"] + dc["mcd"],
            }
            rows.append(row)

        i_lsd = [r["I_lsd"] for r in rows]
        i_mcd = [r["I_mcd"] for r in rows]
        result["codecs"][codec] = {
            "n_pairs": len(rows),
            "summary_lsd": summarize_id(i_lsd),
            "summary_mcd": summarize_id(i_mcd),
            "pairs": rows,
            "top_superadditive_lsd": sorted(rows, key=lambda r: -r["I_lsd"])[:10],
            "top_subadditive_lsd": sorted(rows, key=lambda r: r["I_lsd"])[:10],
            "top_superadditive_mcd": sorted(rows, key=lambda r: -r["I_mcd"])[:10],
            "top_subadditive_mcd": sorted(rows, key=lambda r: r["I_mcd"])[:10],
        }

        s_l = result["codecs"][codec]["summary_lsd"]
        s_m = result["codecs"][codec]["summary_mcd"]
        md += [
            "",
            f"## {codec}",
            "",
            f"共 {len(rows)} 个无序层对。",
            "",
            f"| 指标 | mean \(I_D\) | p50 | p10 | p90 | min | max | \(I_D>0\) 比例 |",
            f"|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| LSD | {s_l['mean']:.4f} | {s_l['p50']:.4f} | {s_l['p10']:.4f} | {s_l['p90']:.4f} | {s_l['min']:.4f} | {s_l['max']:.4f} | {s_l['frac_pos']:.3f} |",
            f"| MCD | {s_m['mean']:.4f} | {s_m['p50']:.4f} | {s_m['p10']:.4f} | {s_m['p90']:.4f} | {s_m['min']:.4f} | {s_m['max']:.4f} | {s_m['frac_pos']:.3f} |",
            "",
            "### LSD 超加性最大的 10 对（\(I_D\) 最大）",
            "",
            "| pair | \(D_{ij}\) | \(D_i\) | \(D_j\) | \(D_{clean}\) | \(I_D\) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in result["codecs"][codec]["top_superadditive_lsd"]:
            md.append(
                f"| {r['pair']} | {r['D_ij_lsd']:.4f} | {r['D_i_lsd']:.4f} | {r['D_j_lsd']:.4f} | {r['D_clean_lsd']:.4f} | {r['I_lsd']:.4f} |"
            )
        md += [
            "",
            "### LSD 亚加性最大的 10 对（\(I_D\) 最小）",
            "",
            "| pair | \(D_{ij}\) | \(D_i\) | \(D_j\) | \(D_{clean}\) | \(I_D\) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in result["codecs"][codec]["top_subadditive_lsd"]:
            md.append(
                f"| {r['pair']} | {r['D_ij_lsd']:.4f} | {r['D_i_lsd']:.4f} | {r['D_j_lsd']:.4f} | {r['D_clean_lsd']:.4f} | {r['I_lsd']:.4f} |"
            )
        md += [
            "",
            "### 全部层对",
            "",
            "| pair | LSD \(I_D\) | MCD \(I_D\) | LSD \(D_{ij}\) | MCD \(D_{ij}\) |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in rows:
            md.append(
                f"| {r['pair']} | {r['I_lsd']:.4f} | {r['I_mcd']:.4f} | {r['D_ij_lsd']:.4f} | {r['D_ij_mcd']:.4f} |"
            )

    out_json = OUT_DIR / "summary_interaction.json"
    out_md = OUT_DIR / "ANALYSIS.md"
    # pairs lists are large; keep full in json
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    for codec in CODECS:
        s = result["codecs"][codec]["summary_lsd"]
        print(
            f"{codec}: n_pairs={s['n']} I_lsd mean={s['mean']:.4f} "
            f"frac_pos={s['frac_pos']:.3f} I_mcd mean={result['codecs'][codec]['summary_mcd']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
