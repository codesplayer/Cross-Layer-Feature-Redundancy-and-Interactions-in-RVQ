"""Latent-Mark 可调用 API（抗 neural codec 水印）。

依赖官方仓库 ``/data1/home/ynian/codes/eval/add_exp/Latent-Mark-main/watermark_research/src/latentmark.py``
（可用 ``LATENTMARK_SRC`` / ``--latentmark-src`` 覆盖）。

典型用法::

  from P1.latentmark_api import LatentMarkAPI, list_audio
  api = LatentMarkAPI(views=["snac_24"], device="cuda")
  api.calibrate(list_audio(calib_dir)[:40])
  wav_wm, sr, info = api.embed_file("a.wav")
  score = api.detect(wav_wm, sr)          # >0 判为有水印
  api.save("latentmark_state.pt")

检测规则（与论文一致）：归一化 margin 的中位数；阈值 0（``score > 0``）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

DEFAULT_LATENTMARK_SRC = Path(r"/data1/home/ynian/codes/eval/add_exp/Latent-Mark-main/watermark_research/src")
_ALT_SRCS = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/Latent-Mark-main/watermark_research/src"),
    Path(r"D:\KimiData\Latent-Mark-main\watermark_research\src"),
    Path(os.environ.get("LATENTMARK_SRC", "") or ""),
)


def resolve_latentmark_src(explicit: str | Path | None = None) -> Path:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    cands.extend([p for p in _ALT_SRCS if str(p)])
    cands.append(DEFAULT_LATENTMARK_SRC)
    for p in cands:
        if p.is_dir() and (p / "latentmark.py").is_file():
            return p.resolve()
    raise FileNotFoundError(
        "Latent-Mark src not found. Set LATENTMARK_SRC or pass latentmark_src=... "
        f"(tried {[str(p) for p in cands]})"
    )


_PATCHED_MODULES: set[int] = set()


def _load_audio_librosa(path: str, seconds: float | None = None) -> tuple[torch.Tensor, int]:
    """读 wav/mp3/flac/ogg（librosa），返回 mono ``(1, T)`` float32。

    替换官方 ``torchaudio.load``：NPU/精简环境下 torchaudio 常读不了 mp3，
    且 Latent-Mark ``calibrate`` 对读失败静默跳过，最终会报 no readable audio。
    """
    import librosa

    dur = float(seconds) if seconds is not None and float(seconds) > 0 else None
    wav, sr = librosa.load(str(path), sr=None, mono=True, duration=dur)
    wav = np.asarray(wav, dtype=np.float32).reshape(1, -1)
    return torch.from_numpy(wav), int(sr)


def _resample_librosa(x: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    """与官方签名一致的 resample。

    - ``sr_in == sr_out``：原样返回（保留梯度，dac_24 matched 嵌印走这条）
    - 需要梯度时优先 ``torchaudio.functional.resample``
    - 否则用 librosa（读文件 / 无梯度推理）
    """
    if int(sr_in) == int(sr_out):
        return x
    if x.requires_grad:
        try:
            import torchaudio

            return torchaudio.functional.resample(x, int(sr_in), int(sr_out))
        except Exception:
            pass
    import librosa

    arr = x.detach().float().cpu().numpy()
    axis = arr.ndim - 1
    out = librosa.resample(arr, orig_sr=int(sr_in), target_sr=int(sr_out), axis=axis)
    y = torch.from_numpy(np.asarray(out, dtype=np.float32))
    return y.to(device=x.device, dtype=x.dtype)


def _patch_snac_local_hub(lm, snac_path: str | Path | None = None) -> None:
    """SNACView 默认拉 hub；有本地目录时改走本地（``SNAC.from_pretrained(dir)``）。"""
    local = str(snac_path or "").strip()
    if not local or not Path(local).is_dir():
        try:
            from base.codec_snac import SNAC_PATH_DEFAULT

            local = str(SNAC_PATH_DEFAULT or "").strip()
        except Exception:
            local = ""
    if not local or not Path(local).is_dir():
        for cand in (
            Path(r"E:\pred_model\snac_24khz"),
            Path("/data1/home/ynian/pred_models/snac_24khz"),
            Path(r"D:\pred_model\snac_24khz"),
        ):
            if cand.is_dir():
                local = str(cand)
                break
    if not local or not Path(local).is_dir():
        return
    hubs = getattr(getattr(lm, "SNACView", None), "HUBS", None)
    if isinstance(hubs, dict) and 24000 in hubs:
        hubs[24000] = local


def set_snac_local_path(snac_path: str | Path, *, latentmark_src: str | Path | None = None) -> None:
    """在构造 LatentMarkAPI 前调用，强制 ``snac_24`` 用指定本地权重。"""
    lm = _ensure_import(resolve_latentmark_src(latentmark_src))
    _patch_snac_local_hub(lm, snac_path)


def _patch_latentmark_io(lm) -> None:
    """把 load_audio / resample 换成 librosa，支持 wav+mp3；并优先本地 SNAC。"""
    mid = id(lm)
    if mid in _PATCHED_MODULES:
        return
    lm.load_audio = _load_audio_librosa
    lm.resample = _resample_librosa
    _patch_snac_local_hub(lm)
    _PATCHED_MODULES.add(mid)


def _ensure_import(src: Path):
    src_s = str(src)
    if src_s not in sys.path:
        sys.path.insert(0, src_s)
    import latentmark as lm  # noqa: WPS433

    _patch_latentmark_io(lm)
    return lm


def list_audio(folder: str | Path, *, recursive: bool = True, limit: int = 0) -> list[str]:
    lm = _ensure_import(resolve_latentmark_src())
    files = lm.list_audio_files(str(folder), recursive=bool(recursive))
    if limit > 0:
        files = files[: int(limit)]
    return files


class LatentMarkAPI:
    """封装 calibrate / embed / detect / save / load。"""

    THRESHOLD = 0.0  # paper: normalized margin > 0

    def __init__(
        self,
        views: Sequence[str] = ("snac_24",),
        device: str = "cuda",
        *,
        axis: str = "cluster",
        k: float = 1.5,
        gamma: float = 1.5,
        steps: int = 150,
        lr: float = 0.005,
        beta: float = 2.5,
        target_sdr: float = 42.0,
        seed: int = 0,
        work_sr: int | None = None,
        latentmark_src: str | Path | None = None,
        **kwargs: Any,
    ):
        self.src = resolve_latentmark_src(latentmark_src)
        self.lm = _ensure_import(self.src)
        self.device = str(device)
        self.views = [str(v) for v in views]
        common = dict(
            k=float(k),
            gamma=float(gamma),
            steps=int(steps),
            lr=float(lr),
            beta=float(beta),
            target_sdr=float(target_sdr),
            seed=int(seed),
            axis=str(axis),
        )
        if work_sr is not None:
            common["work_sr"] = int(work_sr)
        common.update(kwargs)
        self.wm = self.lm.LatentMark(self.views, self.device, **common)

    @property
    def name(self) -> str:
        return str(self.wm.name)

    @property
    def calibrated(self) -> bool:
        return bool(getattr(self.wm, "calib", None)) and len(self.wm.calib) == len(self.wm.views)

    def calibrate(self, files: Sequence[str], seconds: float = 5.0) -> dict:
        files = [str(f) for f in files]
        if not files:
            raise ValueError("calibrate() needs clean audio files")
        self.wm.calibrate(files, seconds=float(seconds))
        return {n: {"mu": c.mu, "sigma": c.sigma, "tau": c.tau, "alpha": c.alpha} for n, c in self.wm.calib.items()}

    def embed(
        self,
        wav: torch.Tensor | np.ndarray,
        sr: int,
    ) -> tuple[torch.Tensor, int, dict]:
        """嵌印。返回 (wav_wm[1,T], sr_out, info)。``sr_out`` 为 Latent-Mark 工作采样率。"""
        if not self.calibrated:
            raise RuntimeError("call calibrate() or load() before embed()")
        x = self._as_tensor(wav)
        out, payload = self.wm.embed(x, int(sr))
        if out.dim() == 1:
            out = out.unsqueeze(0)
        sr_out = int(getattr(self.wm, "wm_sr", sr))
        with torch.no_grad():
            score = float(self.wm.detect(out, sr_out))
        return out.detach().cpu().float(), sr_out, {"payload": payload, "wm_score": score, "method": self.name}

    def embed_file(self, path: str | Path, *, seconds: float | None = None) -> tuple[torch.Tensor, int, dict]:
        wav, sr = self.lm.load_audio(str(path), seconds=seconds)
        wm, sr_out, info = self.embed(wav, sr)
        info["src"] = str(path)
        info["src_sr"] = int(sr)
        return wm, sr_out, info

    def detect(self, wav: torch.Tensor | np.ndarray, sr: int) -> float:
        """返回归一化 margin（中位数）；``> 0`` 判为检出。"""
        if not self.calibrated:
            raise RuntimeError("call calibrate() or load() before detect()")
        x = self._as_tensor(wav)
        return float(self.wm.detect(x, int(sr)))

    def detect_file(self, path: str | Path, *, seconds: float | None = None) -> float:
        wav, sr = self.lm.load_audio(str(path), seconds=seconds)
        return self.detect(wav, sr)

    def margins(self, wav: torch.Tensor | np.ndarray, sr: int) -> dict[str, float]:
        x = self._as_tensor(wav)
        return dict(self.wm.margins(x, int(sr)))

    def is_watermarked(self, wav: torch.Tensor | np.ndarray, sr: int, *, threshold: float | None = None) -> bool:
        thr = self.THRESHOLD if threshold is None else float(threshold)
        return self.detect(wav, sr) > thr

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "views": self.views,
            "device": self.device,
            "name": self.name,
            "axis_kind": self.wm.axis_kind,
            "k": self.wm.k,
            "gamma": self.wm.gamma,
            "steps": self.wm.steps,
            "lr": self.wm.lr,
            "beta": self.wm.beta,
            "target_sdr": self.wm.target_sdr,
            "seed": self.wm.seed,
            "work_sr": self.wm.work_sr,
            "target_mode": self.wm.target_mode,
            "sigma_level": self.wm.sigma_level,
            "hinge": self.wm.hinge,
            "axes": {k: v.detach().cpu() for k, v in self.wm.axes.items()},
            "calib": {
                n: {"mu": c.mu, "sigma": c.sigma, "tau": c.tau, "alpha": c.alpha, "n_frames": c.n_frames, "n_files": c.n_files}
                for n, c in self.wm.calib.items()
            },
            "latentmark_src": str(self.src),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None, *, latentmark_src: str | Path | None = None) -> "LatentMarkAPI":
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        api = cls(
            views=state["views"],
            device=str(device or state.get("device") or "cpu"),
            axis=state.get("axis_kind", "cluster"),
            k=state.get("k", 1.5),
            gamma=state.get("gamma", 1.5),
            steps=state.get("steps", 150),
            lr=state.get("lr", 0.005),
            beta=state.get("beta", 2.5),
            target_sdr=state.get("target_sdr", 42.0),
            seed=state.get("seed", 0),
            work_sr=state.get("work_sr"),
            target_mode=state.get("target_mode"),
            sigma_level=state.get("sigma_level", "file"),
            hinge=state.get("hinge", "mean"),
            latentmark_src=latentmark_src or state.get("latentmark_src"),
        )
        # restore axes + calib (skip re-estimating)
        for n, t in (state.get("axes") or {}).items():
            if n in api.wm.axes:
                api.wm.axes[n] = t.to(api.device)
        ViewCalibration = api.lm.ViewCalibration
        api.wm.calib = {
            n: ViewCalibration(
                float(c["mu"]),
                float(c["sigma"]),
                float(c["tau"]),
                float(c["alpha"]),
                int(c.get("n_frames", 0)),
                int(c.get("n_files", 0)),
            )
            for n, c in (state.get("calib") or {}).items()
        }
        return api

    @staticmethod
    def _as_tensor(wav: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(wav, np.ndarray):
            t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        else:
            t = wav.float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t


def view_roundtrip_attack(
    api: "LatentMarkAPI",
    wav: torch.Tensor | np.ndarray,
    sr: int,
    *,
    view_name: str | None = None,
) -> tuple[torch.Tensor, int]:
    """与官方 ``latentmark.CodecAttack`` 完全一致的 codec round-trip。

    流程：resample→view.sr → ``view.roundtrip`` → resample 回 **输入 sr**。
    检测时请用返回的 ``sr``（通常等于 ``api.wm.wm_sr``）。
    返回 ``wav (1, T)`` float32 CPU tensor 与 ``sr``。
    """
    if not api.calibrated:
        raise RuntimeError("call calibrate() or load() before view_roundtrip_attack()")
    lm = api.lm
    # 优先走官方 CodecAttack，保证与 benchmark.py 一致
    name = str(view_name or (api.views[0] if api.views else ""))
    if not name:
        raise RuntimeError("no view_name / api.views")
    atk = lm.CodecAttack(name, api.device)
    x = api._as_tensor(wav)
    with torch.no_grad():
        y = atk(x, int(sr))  # (1, T) at input sr
    if y.dim() == 1:
        y = y.unsqueeze(0)
    return y.detach().float().cpu(), int(sr)


def codec_roundtrip_attack(
    wav: torch.Tensor | np.ndarray,
    sr: int,
    *,
    codec: str,
    device: str,
    moss_path: str = "",
    dac_path: str = "",
    qwen_path: str = "",
) -> tuple[np.ndarray, int]:
    """用水印仓库自带的 MOSS / DAC / Qwen 做 encode→decode 攻击。

    注意：若评估对象是 Latent-Mark matched DAC，请优先用 ``view_roundtrip_attack``，
    与嵌印 view 完全同一条 ``model.encode/decode`` 路径。

    返回 (wav_out float32 1-D, out_sr)。
    """
    from base.io import match_len
    from base.moss_io import read_wav  # noqa: F401  # keep side imports consistent
    from P2.experiments.token_transition.backends import load_bundle

    codec = str(codec).lower().strip()
    bundle = load_bundle(codec, device, moss_path=moss_path, dac_path=dac_path, qwen_path=qwen_path)
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    # resample to codec sr via librosa
    import librosa

    if int(sr) != int(bundle.sr):
        x = librosa.resample(x, orig_sr=int(sr), target_sr=int(bundle.sr)).astype(np.float32)
    walk = bundle.walk(x)
    codes = np.asarray(walk["codes"], dtype=np.int64)
    y = np.asarray(bundle.decode_from_codes(codes), dtype=np.float32).reshape(-1)
    y = match_len(x, y)
    return y, int(bundle.sr)
