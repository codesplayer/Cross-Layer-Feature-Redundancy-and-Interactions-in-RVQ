"""公共基础设施：MOSS IO、残差、密钥、攻击、克隆器、P2/P3 共享算子。"""

from .attack import add_noise, mp3_roundtrip, resample_roundtrip, stretch
from .config import (
    DEFAULT_WAV_DIR,
    EXP_ROOT,
    KEY,
    NAMES,
    N_NULL,
    NOISE_SNR_DB,
    TEXTS,
    VOXCPM2_PATH,
    WAVLM_SV_PATH,
    default_cluster_dir,
    default_model_path,
    default_table_path,
)
from .detect_shared import (
    LayerNull,
    band_zq_of,
    carrier_stat,
    detect_p1_cosine,
    detect_p2,
    detect_sign,
    feat_cosine,
    frame_cos,
    jaccard_and_agree,
    pearson,
    proj_series,
    tap_stat,
)
from .embed_shared import (
    P2_KINDS,
    P2_POST_MODES,
    embed_carrier,
    embed_carrier_cond,
    embed_p2,
    embed_p2_cond,
    embed_pn,
    embed_sign,
    embed_sign_multilayer,
)
from .io import clean_decode, decode_moss, encode_moss, match_len, read_wav, write_wav_float
from .keys import carrier_w, keyed_bits, pn_seq
from .model import free_gpu, load_model, load_moss
from .trace import encode_hidden, trace_enter
from .exp_parallel import (
    flatten_key,
    launch_clip_workers,
    make_parser,
    merge_dict_lists,
    parse_names_arg,
    resolve_named_clips,
    resolve_out_dir,
)
from .npu_parallel import (
    add_npu_args,
    build_payloads,
    ensure_spawn,
    free_device,
    parse_npu_ids,
    pin_npu,
    run_shards,
)
from .util import dump_json, embed_wav, list_audio

__all__ = [
    "DEFAULT_WAV_DIR", "EXP_ROOT", "KEY", "NAMES", "N_NULL", "NOISE_SNR_DB", "TEXTS",
    "VOXCPM2_PATH", "WAVLM_SV_PATH", "default_cluster_dir", "default_model_path", "default_table_path",
    "load_moss", "load_model", "free_gpu",
    "read_wav", "write_wav_float", "encode_moss", "decode_moss", "match_len", "clean_decode",
    "encode_hidden", "trace_enter",
    "carrier_w", "keyed_bits", "pn_seq",
    "P2_KINDS", "P2_POST_MODES",
    "embed_carrier", "embed_carrier_cond", "embed_p2", "embed_p2_cond",
    "embed_pn", "embed_sign", "embed_sign_multilayer",
    "LayerNull", "band_zq_of", "carrier_stat", "detect_p1_cosine", "detect_p2", "detect_sign",
    "feat_cosine", "frame_cos", "jaccard_and_agree", "pearson", "proj_series", "tap_stat",
    "add_noise", "mp3_roundtrip", "resample_roundtrip", "stretch",
    "dump_json", "embed_wav", "list_audio",
    "add_npu_args", "build_payloads", "ensure_spawn", "free_device",
    "parse_npu_ids", "pin_npu", "run_shards",
    "make_parser", "resolve_out_dir", "parse_names_arg", "resolve_named_clips",
    "launch_clip_workers", "flatten_key", "merge_dict_lists",
]
