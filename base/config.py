"""路径与常量（独立于 codec_an 仓库；数据/权重仍可指向原磁盘路径）。"""

from __future__ import annotations

from pathlib import Path

LOCAL_MODEL = Path(r"/data1/home/ynian/pred_models/MOSS-Audio-Tokenizer")
SERVER_MODEL = Path(r"/data1/home/ynian/pred_models/MOSS-Audio-Tokenizer")
LOCAL_TABLE = Path(r"/data1/home/ynian/codes/codec_an/exp_watermark/codec_an/anas/watermark_rare_tokens/rare_aba.json")
SERVER_TABLE = Path(
    r"/data1/home/ynian/codes/codec_an/exp_watermark/codec_an/anas/watermark_rare_tokens/rare_aba.json"
)
LOCAL_CLUSTER = Path(
    r"/data1/home/ynian/codes/codec_an/exp_watermark/codec_an/codec_plot/embedding_clusters/embedding_clusters_normalize"
)
DEFAULT_WAV_DIR = Path(r"/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500")
LOCAL_MIX_1500 = Path(r"/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500")
DEFAULT_NAMES = ("BAC009S0764W0121", "BAC009S0764W0122", "BAC009S0764W0123")
VOXCPM2_PATH = r"/data1/home/ynian/pred_models/VoxCPM2"
WAVLM_SV_PATH = r"/data1/home/ynian/pred_models/wavlm-base-plus-sv"
EXP_ROOT = Path(r"/data1/home/ynian/codes/eval/add_exp/result")
KEY = 20260928
N_NULL = 32
NOISE_SNR_DB = 20.0
NAMES = tuple(DEFAULT_NAMES)
TEXTS = {
    "BAC009S0764W0121": "甚至出现交易几乎停滞的情况",
    "BAC009S0764W0122": "一二线城市虽然也处于调整中",
    "BAC009S0764W0123": "但因为聚集了过多公共资源",
}
FROZEN_MEAN_T = {
    "aba_embed": 0.3027,
    "slide_v3": 0.3298,
    "wave": 0.7479,
}


def first_existing(*paths) -> str:
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return str(paths[0])


def default_model_path() -> str:
    return first_existing(
        Path(r"e:\pred_model\MOSS\MOSS-Audio-Tokenizer"),
        LOCAL_MODEL,
        SERVER_MODEL,
    )


def default_table_path() -> str:
    return first_existing(
        Path(r"d:\data\codec_an\anas\watermark_rare_tokens\rare_aba.json"),
        LOCAL_TABLE,
        SERVER_TABLE,
    )


def default_cluster_dir() -> str:
    return first_existing(
        LOCAL_CLUSTER,
        Path(r"d:\data\codec_an\codec_plot\embedding_clusters\embedding_clusters_normalize"),
    )


def default_mix_1500() -> str:
    return first_existing(LOCAL_MIX_1500, DEFAULT_WAV_DIR)


def parse_names(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def resolve_wav_paths(wav_dir: Path, names: list[str] | None, n_files: int = 3) -> list[Path]:
    if names:
        out = []
        for name in names:
            stem = Path(name).stem
            d = Path(wav_dir)
            candidates = [d / f"{stem}.wav", d / f"{stem}_orig.wav"]
            p = next((c for c in candidates if c.exists()), None)
            if p is None:
                raise SystemExit(f"missing wav: {candidates[0]}")
            out.append(p)
        return out
    files = sorted(Path(wav_dir).glob("*.wav"))
    if not files:
        raise SystemExit(f"no wav in {wav_dir}")
    return files[: int(n_files)]
