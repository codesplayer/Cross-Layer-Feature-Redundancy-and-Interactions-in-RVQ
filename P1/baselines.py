"""AudioSeal / WavMark 轻量封装（可选依赖；缺包时报清晰错误）。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

AS_GEN_CANDIDATES = (
    Path("/data1/home/ynian/pred_models/audioseal/generator_base.pth"),
    Path(r"E:\pred_model\audioseal\generator_base.pth"),
)
AS_DET_CANDIDATES = (
    Path("/data1/home/ynian/pred_models/audioseal/detector_base.pth"),
    Path(r"E:\pred_model\audioseal\detector_base.pth"),
)
WM_CKPT_NAME = "step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl"
WM_CKPT_CANDIDATES = (
    Path("/data1/home/ynian/pred_models/WavMark") / WM_CKPT_NAME,
    Path("/home/ma-user/work/ynian/predmodel/WavMark") / WM_CKPT_NAME,
    Path(r"E:\pred_model\wavmark") / WM_CKPT_NAME,
)
PAYLOAD_BITS = 16
PATTERN_BITS = 16


def first_file(*cands: Path) -> str:
    for p in cands:
        if p.is_file():
            return str(p)
    return str(cands[0])


def disable_torch_compile_on_npu(device: str) -> None:
    """AudioSeal/moshi 默认 torch.compile→Inductor；NPU 上会 AssertionError: Device npu not supported。"""
    if not str(device).startswith("npu"):
        return
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    try:
        import torch._dynamo as dynamo

        dynamo.config.disable = True
        dynamo.config.suppress_errors = True
    except Exception:
        pass

    def _identity_compile(model=None, *args, **kwargs):
        if model is None:

            def _deco(fn):
                return fn

            return _deco
        return model

    try:
        torch.compile = _identity_compile  # type: ignore[assignment]
    except Exception:
        pass


def _npu_conv_transpose_too_wide(conv: torch.nn.ConvTranspose1d) -> bool:
    """Ascend: (K-1)*dilation - padding <= 255，否则 aclnnConvolution 报 161002。"""
    k = int(conv.weight.shape[-1])
    dil = int(conv.dilation[0])
    pad = int(conv.padding[0])
    return (k - 1) * dil - pad > 255


def _as_nonoverlap_transpose(conv: torch.nn.ConvTranspose1d) -> bool:
    """stride==kernel、padding=0 时转置卷积等于无重叠逐帧展开。"""
    k = int(conv.weight.shape[-1])
    return (
        conv.groups == 1
        and int(conv.stride[0]) == k
        and int(conv.padding[0]) == 0
        and int(conv.dilation[0]) == 1
        and int(conv.output_padding[0]) == 0
    )


def patch_npu_large_conv_transpose(root: torch.nn.Module) -> list[str]:
    """检测器 reverse_convolution 核宽 320 > NPU 上限 255；stride 同为 320，用 einsum 等价替换。"""
    patched: list[str] = []
    for name, mod in list(root.named_modules()):
        if not isinstance(mod, torch.nn.ConvTranspose1d):
            continue
        if not _npu_conv_transpose_too_wide(mod):
            continue
        if not _as_nonoverlap_transpose(mod):
            raise RuntimeError(
                f"{name}: ConvTranspose1d kernel={tuple(mod.weight.shape)} "
                "超过 NPU 上限，且不是 stride==kernel 的无重叠展开，无法替换"
            )

        def _forward(x: torch.Tensor, _conv: torch.nn.ConvTranspose1d = mod) -> torch.Tensor:
            y = torch.einsum("nct,cok->notk", x, _conv.weight)
            y = y.reshape(y.shape[0], y.shape[1], -1)
            if _conv.bias is not None:
                y = y + _conv.bias.view(1, -1, 1)
            return y

        mod.forward = _forward  # type: ignore[method-assign]
        patched.append(name)
    return patched


def load_audioseal(generator: str, detector: str, device: str, nbits: int = 16):
    disable_torch_compile_on_npu(device)
    try:
        from audioseal import AudioSeal
    except ImportError as e:
        raise SystemExit("未安装 audioseal：pip install audioseal\n" + str(e)) from e
    gen = AudioSeal.load_generator(generator, nbits=int(nbits)).to(device).eval()
    det = AudioSeal.load_detector(detector, nbits=int(nbits)).to(device).eval()
    if str(device).startswith("npu"):
        for m in (gen, det):
            patch_npu_large_conv_transpose(m)
    return gen, det


@torch.no_grad()
def audioseal_embed(gen, wav: np.ndarray, sr: int, device, msg: torch.Tensor | None = None):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    x = torch.from_numpy(wav).view(1, 1, -1).to(device)
    if msg is None:
        nbits = int(gen.msg_processor.nbits)
        msg = torch.randint(0, 2, (1, nbits), device=device)
    else:
        msg = msg.to(device)
        if msg.dim() == 1:
            msg = msg.unsqueeze(0)
    wm = x + gen.get_watermark(x, int(sr), message=msg)
    return wm[0, 0].detach().float().cpu().numpy().astype(np.float32), msg.detach().cpu()


@torch.no_grad()
def audioseal_detect(det, wav: np.ndarray, sr: int, device) -> float:
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32).reshape(-1)).view(1, 1, -1).to(device)
    p, _ = det.detect_watermark(x, int(sr))
    return float(p.reshape(-1)[0].item() if torch.is_tensor(p) else p)


def patch_wavmark_stft_contiguous(model) -> None:
    """NPU 上 view_as_complex / stft 要求末维 stride=1；WavMark 原版未保证 contiguous。"""
    if getattr(model, "_wm_stft_patched", False):
        return
    _stft, _istft = model.stft, model.istft
    _encode, _decode = model.encode, model.decode

    def stft(data):
        return _stft(data.reshape(data.shape[0], -1).contiguous())

    def istft(signal_wmd_fft):
        return _istft(signal_wmd_fft.contiguous())

    def encode(signal, message):
        return _encode(signal.contiguous(), message.contiguous())

    def decode(signal):
        return _decode(signal.contiguous())

    model.stft = stft
    model.istft = istft
    model.encode = encode
    model.decode = decode
    model._wm_stft_patched = True


def load_wavmark(checkpoint: str, device: str):
    try:
        import wavmark
    except ImportError as e:
        raise SystemExit("未安装 wavmark：pip install wavmark\n" + str(e)) from e
    model = wavmark.load_model(str(checkpoint))
    model = model.to(device).eval()
    patch_wavmark_stft_contiguous(model)
    return model


def _to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if int(sr) == 16000:
        return wav
    return librosa.resample(wav, orig_sr=int(sr), target_sr=16000).astype(np.float32)


def _from_16k(wav16: np.ndarray, sr: int, n_ref: int) -> np.ndarray:
    import librosa

    wav16 = np.asarray(wav16, dtype=np.float32).reshape(-1)
    if int(sr) == 16000:
        out = wav16
    else:
        out = librosa.resample(wav16, orig_sr=16000, target_sr=int(sr)).astype(np.float32)
    if out.shape[0] >= n_ref:
        return out[:n_ref]
    return np.pad(out, (0, n_ref - out.shape[0])).astype(np.float32)


@torch.no_grad()
def wavmark_embed(model, wav: np.ndarray, sr: int, payload: np.ndarray | None = None):
    import wavmark

    wav = np.ascontiguousarray(np.asarray(wav, dtype=np.float32).reshape(-1))
    wav16 = np.ascontiguousarray(_to_16k(wav, sr))
    if payload is None:
        payload = np.random.randint(0, 2, size=(PAYLOAD_BITS,), dtype=np.int64)
    payload = np.asarray(payload, dtype=np.int64).reshape(-1)
    wm16, info = wavmark.encode_watermark(
        model,
        wav16,
        payload,
        pattern_bit_length=PATTERN_BITS,
        show_progress=False,
    )
    wm = _from_16k(wm16, sr, wav.shape[0])
    return np.ascontiguousarray(wm.astype(np.float32)), payload, dict(info or {})


@torch.no_grad()
def wavmark_detect(model, wav: np.ndarray, sr: int, payload_tgt: np.ndarray | None = None) -> dict:
    import wavmark

    wav16 = np.ascontiguousarray(_to_16k(wav, sr))
    decoded, info = wavmark.decode_watermark(
        model,
        wav16,
        decode_batch_size=10,
        len_start_bit=PATTERN_BITS,
        show_progress=False,
    )
    info = dict(info or {})
    detected = decoded is not None
    bit_acc = float("nan")
    if decoded is not None and payload_tgt is not None:
        p = np.asarray(decoded, dtype=np.int64).reshape(-1)
        t = np.asarray(payload_tgt, dtype=np.int64).reshape(-1)
        n = min(p.size, t.size)
        bit_acc = float(np.mean(p[:n] == t[:n])) if n else float("nan")
    return {
        "wm_prob": 1.0 if detected else 0.0,
        "detected": bool(detected),
        "bit_acc": bit_acc,
        "n_pattern_hits": int(len(info.get("results") or [])),
    }
