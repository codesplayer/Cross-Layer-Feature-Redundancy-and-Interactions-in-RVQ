"""MOSS 大规模层替换 + 后层补偿（单层 / 多层）· 8 进程 × 8 NPU。

替换方式对齐 ``MOSS_replace_layer0.py`` / ``MOSS_tokenizer.py`` 与 DNSMOS 图：

- **replace_one**：每次只随机替换一层 L，其余冻结（图 replace_one）
- **replace_all**：从最深层起累计随机替换 k 层，浅层保留（图 replace_all）

补偿对齐 ``watermark.band`` 的 ``compensate_post_layers``：

- **replace_one + comp**：固定 0..L（含被替换的 L），残差减去其 z_q，再量化 L+1..
- **replace_all + comp**：固定浅层 0..(n-k-1)，对深层 (n-k)..(n-1) 按残差重新量化
  （相对「深层随机码」的正确残差填充）

数据默认：``/home/ma-user/work/ynian/codec_an/audios/data/sub_audio``（中英约 1000 条）。
输出目录下同时保存 nocomp / comp 波形。

例（8 NPU）::

  python codecs/MOSS_replace_compensate_mp.py \\
    --mode both --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7

单卡调试::

  python codecs/MOSS_replace_compensate_mp.py --mode one --num-workers 1 --device npu:0 --max-files 8
"""

from __future__ import annotations

import argparse
import os
import random
import traceback
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

# ---------------------------------------------------------------------------
# 默认路径（集群）
# ---------------------------------------------------------------------------
DEFAULT_AUDIO_DIR = "/home/ma-user/work/ynian/codec_an/audios/data/sub_audio"
DEFAULT_MODEL = "/home/ma-user/work/ynian/code/codec_an/model/MOSS-Audio-Tokenizer"
DEFAULT_OUT = "/home/ma-user/work/ynian/codec_an/audios/speech/MOSS/replace_comp"


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_moss(device: str, model_path: str):
    if str(device).startswith("npu"):
        import torch_npu  # noqa: F401

        if ":" in device:
            torch.npu.set_device(int(device.split(":", 1)[1]))
    model = AutoModel.from_pretrained(
        model_path, device_map=device, trust_remote_code=True
    ).eval()
    return model


def encode_moss(model, audio_path: str, device: str) -> torch.Tensor:
    wav, _sr = librosa.load(audio_path, sr=model.sampling_rate)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.encode(wav_t)
    return out.audio_codes.squeeze(1).long()  # [N, T]


def decode_moss(model, tokens: torch.Tensor) -> np.ndarray:
    codes = tokens.unsqueeze(1)  # [N, 1, T]
    with torch.no_grad():
        out = model.decode(codes)
    return out.audio.squeeze(0).float().cpu().numpy()


def wav_to_mono(wav: np.ndarray) -> np.ndarray:
    w = np.asarray(wav, dtype=np.float32)
    if w.ndim > 1:
        w = w.reshape(w.shape[0], -1).mean(axis=0)
    return w.reshape(-1)


def codebook_size_of(model) -> int:
    q0 = model.quantizer.quantizers[0]
    for attr in ("codebook_size", "n_codes", "num_embeddings"):
        if hasattr(q0, attr):
            return int(getattr(q0, attr))
    emb = getattr(q0, "embedding", None) or getattr(q0, "codebook", None)
    if emb is not None and hasattr(emb, "weight"):
        return int(emb.weight.shape[0])
    return 1024


def perturb_one_layer(
    tokens: torch.Tensor, layer_idx: int, codebook_size: int
) -> torch.Tensor:
    """只随机替换 layer_idx，其余不变。"""
    out = tokens.clone()
    T = out.shape[1]
    out[layer_idx, :] = torch.randint(
        0, codebook_size, (T,), device=tokens.device, dtype=torch.long
    )
    return out


def perturb_deep_layers(
    tokens: torch.Tensor, num_replace: int, codebook_size: int
) -> torch.Tensor:
    """从最深层起随机替换 num_replace 层（浅层保留）。同 MOSS_tokenizer.perturb_layer。"""
    out = tokens.clone()
    n, T = out.shape
    k = max(0, min(int(num_replace), n))
    for i in range(k):
        layer_idx = n - 1 - i
        out[layer_idx, :] = torch.randint(
            0, codebook_size, (T,), device=tokens.device, dtype=torch.long
        )
    return out


def _layer_zq_from_ids(q, ids_1t: torch.Tensor) -> torch.Tensor:
    z = q.decode_code(ids_1t).float()
    weight = getattr(getattr(q, "out_proj", None), "weight", None)
    if weight is not None and int(z.shape[1]) == int(weight.shape[1]):
        z = q.out_proj(z).float()
    return z


def _zq_full(q, ids_1t: torch.Tensor, max_time: int, mask: torch.Tensor) -> torch.Tensor:
    z_q = _layer_zq_from_ids(q, ids_1t)
    if z_q.shape[-1] < max_time:
        pad = torch.zeros(
            (1, z_q.shape[1], max_time - z_q.shape[-1]),
            device=z_q.device,
            dtype=z_q.dtype,
        )
        z_q_full = torch.cat([z_q, pad], dim=-1)
    else:
        z_q_full = z_q[..., :max_time]
    return z_q_full * mask.unsqueeze(1)


def encode_hidden(model, wav_np: np.ndarray, device: str):
    wav = (
        torch.from_numpy(np.ascontiguousarray(wav_np))
        .float()
        .view(1, 1, -1)
        .to(device)
    )
    with torch.no_grad():
        out = model.encode(wav, return_dict=True)
    if isinstance(out, (tuple, list)):
        raise RuntimeError("encode(..., return_dict=True) 未返回 dict")
    e = out.encoder_hidden_states
    if e is None:
        raise RuntimeError("encoder_hidden_states is None")
    lengths = out.audio_codes_lengths
    if lengths is None:
        lengths = torch.full((e.shape[0],), e.shape[-1], device=e.device, dtype=torch.long)
    return e, lengths


@torch.no_grad()
def compensate_fix_prefix(
    model,
    wav_np: np.ndarray,
    mod_codes: torch.Tensor | np.ndarray,
    *,
    fix_end: int,
    subtract_fix: bool = True,
    device: str,
) -> tuple[np.ndarray, torch.Tensor]:
    """固定前缀层 0..fix_end，对后续层按残差重新量化。

    ``subtract_fix=True``（推荐）：残差减去固定前缀（含 fix_end）的 z_q。
    ``fix_end=-1``：不固定任何层，整段重新量化。
    """
    if isinstance(mod_codes, np.ndarray):
        mod = torch.from_numpy(np.ascontiguousarray(mod_codes)).long().to(device)
    else:
        mod = mod_codes.long().to(device)

    hidden, lengths = encode_hidden(model, wav_np, device)
    quantizer = model.quantizer
    n_q = len(quantizer.quantizers)
    z = quantizer.input_proj(hidden)
    batch, _, max_time = z.shape
    t_codes = int(mod.shape[1])
    t_use = min(int(lengths[0].item()), t_codes, int(max_time))
    mask = torch.arange(max_time, device=hidden.device).expand(batch, max_time) < t_use
    residual = z.clone().float()
    new_codes = mod.clone()
    peel_end = int(fix_end) if subtract_fix else int(fix_end) - 1

    for i in range(n_q):
        masked = residual * mask.unsqueeze(1)
        q = quantizer.quantizers[i]
        if i <= peel_end:
            ids = new_codes[i, :t_use].view(1, t_use)
            residual = residual - _zq_full(q, ids, max_time, mask)
        else:
            z_q_i, indices, _ = q(masked)
            new_codes[i, :t_use] = indices[0, :t_use].long()
            residual = residual - z_q_i * mask.unsqueeze(1)

    wav_out = decode_moss(model, new_codes)
    return wav_to_mono(wav_out), new_codes


def list_audio(audio_dir: str, max_files: int | None) -> list[str]:
    names = sorted(
        n
        for n in os.listdir(audio_dir)
        if os.path.isfile(os.path.join(audio_dir, n))
        and n.lower().endswith((".wav", ".flac", ".mp3", ".ogg"))
    )
    if not names:
        # 兼容无扩展名过滤（与参考脚本一致）
        names = sorted(
            n
            for n in os.listdir(audio_dir)
            if os.path.isfile(os.path.join(audio_dir, n))
        )
    if max_files is not None and max_files > 0:
        names = names[: int(max_files)]
    return names


def save_wav(path: str, wav: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav_to_mono(wav), sr)


def process_one_audio(
    *,
    model,
    device: str,
    audio_dir: str,
    audio_name: str,
    out_root: str,
    modes: set[str],
    do_comp: bool,
    do_nocomp: bool,
    codebook_size: int,
    resume: bool,
) -> dict:
    audio_path = os.path.join(audio_dir, audio_name)
    wav_np, _ = librosa.load(audio_path, sr=model.sampling_rate)
    wav_np = wav_to_mono(np.asarray(wav_np, dtype=np.float32))
    tokens = encode_moss(model, audio_path, device)
    n = int(tokens.size(0))
    sr = int(model.sampling_rate)
    n_saved = 0
    n_skip = 0

    if "one" in modes:
        for layer_idx in range(n):
            mod = perturb_one_layer(tokens, layer_idx, codebook_size)
            if do_nocomp:
                d = os.path.join(out_root, "replace_one", "nocomp", f"rp_one_{layer_idx}")
                fp = os.path.join(d, audio_name)
                if resume and os.path.isfile(fp):
                    n_skip += 1
                else:
                    wav = decode_moss(model, mod)
                    save_wav(fp, wav, sr)
                    n_saved += 1
            if do_comp:
                d = os.path.join(out_root, "replace_one", "comp_sub", f"rp_one_{layer_idx}")
                fp = os.path.join(d, audio_name)
                if resume and os.path.isfile(fp):
                    n_skip += 1
                else:
                    wav_c, _ = compensate_fix_prefix(
                        model,
                        wav_np,
                        mod,
                        fix_end=layer_idx,
                        subtract_fix=True,
                        device=device,
                    )
                    save_wav(fp, wav_c, sr)
                    n_saved += 1

    if "all" in modes:
        for num_replace in range(n, 0, -1):
            mod = perturb_deep_layers(tokens, num_replace, codebook_size)
            if do_nocomp:
                d = os.path.join(out_root, "replace_all", "nocomp", f"rp_all_{num_replace}")
                fp = os.path.join(d, audio_name)
                if resume and os.path.isfile(fp):
                    n_skip += 1
                else:
                    wav = decode_moss(model, mod)
                    save_wav(fp, wav, sr)
                    n_saved += 1
            if do_comp:
                # 浅层保留原码；深层按残差重量化（fix_end = 最后一个未替换层）
                fix_end = n - num_replace - 1
                d = os.path.join(out_root, "replace_all", "comp_resynth", f"rp_all_{num_replace}")
                fp = os.path.join(d, audio_name)
                if resume and os.path.isfile(fp):
                    n_skip += 1
                else:
                    wav_c, _ = compensate_fix_prefix(
                        model,
                        wav_np,
                        mod,
                        fix_end=fix_end,
                        subtract_fix=True,
                        device=device,
                    )
                    save_wav(fp, wav_c, sr)
                    n_saved += 1

    return {
        "name": audio_name,
        "n_layers": n,
        "n_saved": n_saved,
        "n_skip": n_skip,
        "ok": True,
    }


def _worker(payload: dict) -> dict:
    wid = int(payload["worker_id"])
    device = str(payload["device"])
    names = list(payload["audio_names"])
    set_seeds(int(payload.get("seed", 42)) + wid * 10007)

    # 限制 CPU 线程，避免 8 进程抢核
    try:
        torch.set_num_threads(int(payload.get("cpu_threads") or 1))
    except Exception:
        pass

    print(f"[w{wid}] load model on {device} files={len(names)}", flush=True)
    model = load_moss(device, payload["model_path"])
    cbs = codebook_size_of(model)
    modes = set(payload["modes"])
    rows = []
    n_err = 0
    for i, name in enumerate(names, 1):
        print(f"[w{wid}] {i}/{len(names)} {name}", flush=True)
        try:
            row = process_one_audio(
                model=model,
                device=device,
                audio_dir=payload["audio_dir"],
                audio_name=name,
                out_root=payload["out_dir"],
                modes=modes,
                do_comp=bool(payload["do_comp"]),
                do_nocomp=bool(payload["do_nocomp"]),
                codebook_size=cbs,
                resume=bool(payload.get("resume")),
            )
            rows.append(row)
        except Exception as exc:
            n_err += 1
            err = f"{type(exc).__name__}: {exc}"
            print(f"[w{wid}] FAIL {name}: {err}", flush=True)
            traceback.print_exc()
            rows.append({"name": name, "ok": False, "error": err})
    return {
        "worker_id": wid,
        "device": device,
        "n_files": len(names),
        "n_error": n_err,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOSS replace_one/all + compensate · multi-NPU")
    p.add_argument("--audio-dir", type=str, default=DEFAULT_AUDIO_DIR)
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    p.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=("one", "all", "both"),
        help="one=单层替换; all=多层累计; both=两者都做",
    )
    p.add_argument(
        "--variants",
        type=str,
        default="comp",
        help="逗号分隔: nocomp,comp（comp=补偿）",
    )
    p.add_argument("--max-files", type=int, default=1000, help="最多处理条数，默认 1000")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--npu-ids", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--device", type=str, default="", help="单进程时设备，如 npu:0")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="已存在 wav 则跳过")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audio_dir = str(args.audio_dir)
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(audio_dir):
        raise SystemExit(f"audio dir not found: {audio_dir}")
    if not os.path.isdir(args.model_path):
        raise SystemExit(f"model path not found: {args.model_path}")

    names = list_audio(audio_dir, int(args.max_files) if args.max_files > 0 else None)
    if not names:
        raise SystemExit(f"no audio in {audio_dir}")

    mode = str(args.mode)
    modes = {"one", "all"} if mode == "both" else {mode}
    variants = {v.strip() for v in str(args.variants).split(",") if v.strip()}
    do_nocomp = "nocomp" in variants
    do_comp = "comp" in variants
    if not do_nocomp and not do_comp:
        raise SystemExit("--variants 至少包含 nocomp 或 comp")

    n_workers = max(1, int(args.num_workers))
    npu_ids = [int(x.strip()) for x in str(args.npu_ids).split(",") if x.strip()]
    if n_workers > 1 and len(npu_ids) < n_workers:
        raise SystemExit(f"need >= {n_workers} npu ids, got {npu_ids}")

    print(
        f"MOSS replace+comp files={len(names)} modes={sorted(modes)} "
        f"variants={sorted(variants)} workers={n_workers} -> {out_dir}",
        flush=True,
    )

    common = {
        "model_path": str(args.model_path),
        "audio_dir": audio_dir,
        "out_dir": out_dir,
        "modes": sorted(modes),
        "do_comp": do_comp,
        "do_nocomp": do_nocomp,
        "seed": int(args.seed),
        "cpu_threads": int(args.cpu_threads),
        "resume": bool(args.resume),
    }

    if n_workers == 1:
        device = (args.device or "").strip() or (
            f"npu:{npu_ids[0]}" if npu_ids else "npu:0"
        )
        payloads = [
            {
                "worker_id": 0,
                "device": device,
                "audio_names": names,
                **common,
            }
        ]
    else:
        shards: list[list[str]] = [[] for _ in range(n_workers)]
        for i, name in enumerate(names):
            shards[i % n_workers].append(name)
        payloads = []
        for wid, shard in enumerate(shards):
            print(f"  worker {wid} -> npu:{npu_ids[wid]} files={len(shard)}", flush=True)
            payloads.append(
                {
                    "worker_id": wid,
                    "device": f"npu:{npu_ids[wid]}",
                    "audio_names": shard,
                    **common,
                }
            )

    all_rows: list[dict] = []
    n_err = 0
    if n_workers == 1:
        out = _worker(payloads[0])
        all_rows.extend(out["rows"])
        n_err += int(out["n_error"])
    else:
        # spawn 上下文：避免 fork 后再初始化 NPU/torch
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_worker, payloads)
        for out in results:
            print(
                f"[main] worker {out['worker_id']} done "
                f"files={out['n_files']} err={out['n_error']}",
                flush=True,
            )
            all_rows.extend(out["rows"])
            n_err += int(out["n_error"])

    ok = sum(1 for r in all_rows if r.get("ok"))
    meta = {
        "audio_dir": audio_dir,
        "model_path": str(args.model_path),
        "out_dir": out_dir,
        "n_files": len(names),
        "n_ok": ok,
        "n_error": n_err,
        "modes": sorted(modes),
        "variants": sorted(variants),
        "num_workers": n_workers,
        "npu_ids": npu_ids if n_workers > 1 else [],
        "layout": {
            "replace_one/nocomp/rp_one_{L}/": "单层随机替换，后层冻结",
            "replace_one/comp_sub/rp_one_{L}/": "单层替换 + 固定0..L后量化L+1..",
            "replace_all/nocomp/rp_all_{k}/": "深层累计随机替换 k 层",
            "replace_all/comp_resynth/rp_all_{k}/": "固定浅层后按残差重量化深层",
        },
    }
    meta_path = Path(out_dir) / "run_meta.json"
    meta_path.write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"done ok={ok}/{len(names)} err={n_err} meta={meta_path}", flush=True)


if __name__ == "__main__":
    main()
