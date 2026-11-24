"""汇总 rvq_extra_eta_n1500 与 rvq_af_n1500。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ETA = Path(r"E:\code\water_marks\result\rvq\rvq_extra_eta_n1500")
AF = Path(r"E:\code\water_marks\result\rvq\rvq_af_n1500")
OUT = Path(r"E:\code\water_marks\result\rvq\RVQ_n1500_统计.md")
CODECS = ("moss", "dac", "qwen")


def load_dir(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for p in path.glob("*.json"):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def arr(rows, key):
    vals = []
    for r in rows:
        v = r.get(key)
        if v is None:
            vals.append(np.nan)
        else:
            vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def desc(x: np.ndarray) -> dict:
    v = x[np.isfinite(x)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p10": float(np.percentile(v, 10)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
    }


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return None, None
    xx, yy = x[m], y[m]
    pear = float(np.corrcoef(xx, yy)[0, 1])
    rx = xx.argsort().argsort().astype(np.float64)
    ry = yy.argsort().argsort().astype(np.float64)
    spear = float(np.corrcoef(rx, ry)[0, 1])
    return pear, spear


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def chain_field(rows, chain_name, key):
    """A/C 的 chains[].key，key 可以是 dict 里的 mean。"""
    vals = []
    for r in rows:
        for c in r.get("chains") or []:
            if chain_name and c.get("chain") != chain_name:
                continue
            v = c.get(key)
            if isinstance(v, dict):
                v = v.get("mean")
            if v is not None:
                vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def main():
    lines = ["# RVQ n=1500 统计\n\n", "数据目录：`result/rvq/rvq_extra_eta_n1500` 与 `result/rvq/rvq_af_n1500`。每条语音一个 JSON。均值、分位数都在句子之间计算。\n\n"]

    lines.append("## 1. 额外方向残留与失真\n\n")
    lines.append("| codec | n | η_extra | η_initial | C⊥ | LSD | MCD | Pearson η–LSD | Spearman η–LSD | Pearson η–MCD | Spearman η–MCD |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    eta_detail = {}
    for codec in CODECS:
        rows = load_dir(ETA / codec)
        eta = arr(rows, "eta_extra")
        ini = arr(rows, "eta_initial")
        can = arr(rows, "cancel_ratio")
        lsd = arr(rows, "lsd")
        mcd = arr(rows, "mcd")
        modes = {}
        for r in rows:
            modes[r.get("mcd_mode") or "missing"] = modes.get(r.get("mcd_mode") or "missing", 0) + 1
        pl, sl = corr(eta, lsd)
        pm, sm = corr(eta, mcd)
        de, di, dc, dl, dm = desc(eta), desc(ini), desc(can), desc(lsd), desc(mcd)
        lines.append(
            f"| {codec} | {de.get('n', 0)} | {fmt(de.get('mean'))} | {fmt(di.get('mean'))} | {fmt(dc.get('mean'))} | "
            f"{fmt(dl.get('mean'))} | {fmt(dm.get('mean'))} | {fmt(pl)} | {fmt(sl)} | {fmt(pm)} | {fmt(sm)} |\n"
        )
        eta_detail[codec] = {"eta": de, "lsd": dl, "mcd": dm, "modes": modes, "chains": {}}
        # qwen chains
        names = sorted({c.get("chain") for r in rows for c in (r.get("chains") or []) if c.get("chain")})
        if len(names) > 1:
            for name in names:
                ev, lv = [], []
                for r in rows:
                    hit = next((c for c in r.get("chains") or [] if c.get("chain") == name), None)
                    if hit:
                        ev.append(hit.get("eta_extra"))
                eta_detail[codec]["chains"][name] = desc(np.asarray(ev, dtype=np.float64))
    lines.append("\n")
    for codec, info in eta_detail.items():
        e = info["eta"]
        lines.append(
            f"- {codec}：η_extra p10/p50/p90 = {fmt(e.get('p10'))}/{fmt(e.get('p50'))}/{fmt(e.get('p90'))}，"
            f"LSD {fmt(info['lsd'].get('p50'))}，MCD {fmt(info['mcd'].get('p50'))}，MCD 口径 {info['modes']}\n"
        )
        for name, d in info["chains"].items():
            lines.append(f"  - {codec}/{name} η_extra 均值 {fmt(d.get('mean'))}，n={d.get('n')}\n")

    lines.append("\n## 2. 实验 A：能量分解\n\n")
    lines.append("表内是各句逐帧比值平均之后，再对句子取均值。\n\n")
    lines.append("| codec | chain | n | E1 | E2 | E3 | Ecross | Epos | Eneg | 2⟨Q,r⟩ |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for codec in CODECS:
        rows = load_dir(AF / "A" / codec)
        names = sorted({c.get("chain") for r in rows for c in r.get("chains") or []})
        for name in names:
            stats = {}
            for key in ("E1", "E2", "E3", "Ecross", "Epos", "Eneg", "Eqr"):
                stats[key] = desc(chain_field(rows, name, key))
            n = stats["E1"].get("n", 0)
            lines.append(
                f"| {codec} | {name} | {n} | {fmt(stats['E1'].get('mean'))} | {fmt(stats['E2'].get('mean'))} | "
                f"{fmt(stats['E3'].get('mean'))} | {fmt(stats['Ecross'].get('mean'))} | {fmt(stats['Epos'].get('mean'))} | "
                f"{fmt(stats['Eneg'].get('mean'))} | {fmt(stats['Eqr'].get('mean'))} |\n"
            )

    lines.append("\n## 3. 实验 C：正交抵消\n\n")
    lines.append("| codec | chain | n | C⊥ | 打乱后 C⊥ |\n|---|---|---:|---:|---:|\n")
    for codec in CODECS:
        rows = load_dir(AF / "C" / codec)
        names = sorted({c.get("chain") for r in rows for c in r.get("chains") or []})
        for name in names:
            c1 = desc(chain_field(rows, name, "cperp_mean"))
            c2 = desc(chain_field(rows, name, "cperp_shuffle_mean"))
            lines.append(f"| {codec} | {name} | {c1.get('n', 0)} | {fmt(c1.get('mean'))} | {fmt(c2.get('mean'))} |\n")

    lines.append("\n## 4. 实验 D：残差能量还在下降的层\n\n")
    lines.append("表内是该层使残差能量下降的帧比例，对句子取均值。\n\n")
    lines.append("| codec | chain | L0 | L2 | L7 | L15 | 末层 |\n|---|---|---:|---:|---:|---:|---:|\n")
    for codec in CODECS:
        rows = load_dir(AF / "D" / codec)
        names = sorted({c.get("chain") for r in rows for c in r.get("chains") or []})
        for name in names:
            series = []
            for r in rows:
                hit = next((c for c in r.get("chains") or [] if c.get("chain") == name), None)
                if not hit:
                    continue
                layers = hit.get("layers") or []
                series.append([float(x["dE_pos_frac"]) for x in layers])
            if not series:
                continue
            m = np.mean(np.asarray(series), axis=0)
            last = int(m.shape[0] - 1)
            def at(i):
                return fmt(m[i]) if i < m.shape[0] else "—"
            lines.append(f"| {codec} | {name} | {at(0)} | {at(2)} | {at(7)} | {at(15)} | {at(last)} |\n")

    lines.append("\n## 5. 实验 E：固定后层与后层重算\n\n")
    lines.append("沿本层码方向、幅度 0.5。波形相关和 LSD 相对未扰动重构，对 1500 句取均值。\n\n")
    lines.append("| codec | L | mode | 波形相关 | LSD | ‖ΔQ‖/‖Q‖ |\n|---|---:|---|---:|---:|---:|\n")
    for codec in CODECS:
        rows = load_dir(AF / "E" / codec)
        bucket = {}
        for r in rows:
            for item in r.get("rows") or []:
                if item.get("direction") not in (None, "along_q"):
                    continue
                if float(item.get("alpha", -1)) != 0.5:
                    continue
                key = (int(item["layer"]), item["mode"])
                bucket.setdefault(key, []).append(item)
        for key in sorted(bucket):
            xs = bucket[key]
            wave = np.mean([x["wave_corr"] for x in xs if x.get("wave_corr") is not None])
            lsd = np.mean([x["lsd"] for x in xs if x.get("lsd") is not None])
            rep = [x.get("repr_ratio") for x in xs if x.get("repr_ratio") is not None]
            rep_s = fmt(float(np.mean(rep))) if rep else "—"
            lines.append(f"| {codec} | {key[0]} | {key[1]} | {fmt(wave)} | {fmt(lsd)} | {rep_s} |\n")

    lines.append("\n## 6. 实验 F：译码器方向敏感度\n\n")
    lines.append("幅度 0.25。Qwen 的 F 没有连续解码，未列入。\n\n")
    lines.append("| codec | direction | sign | 波形相关 | LSD | n |\n|---|---|---:|---:|---:|---:|\n")
    for codec in ("moss", "dac"):
        rows = load_dir(AF / "F" / codec)
        bucket = {}
        for r in rows:
            for item in r.get("rows") or []:
                if float(item.get("scale", -1)) != 0.25:
                    continue
                key = (item["direction"], int(item["sign"]))
                bucket.setdefault(key, []).append(item)
        for key in sorted(bucket):
            xs = bucket[key]
            lines.append(
                f"| {codec} | {key[0]} | {key[1]} | {fmt(np.mean([x['wave_corr'] for x in xs]))} | "
                f"{fmt(np.mean([x['lsd'] for x in xs]))} | {len(xs)} |\n"
            )

    OUT.write_text("".join(lines), encoding="utf-8")
    print(OUT)
    print("".join(lines[:40]))


if __name__ == "__main__":
    main()
