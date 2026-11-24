"""RVQ helpers previously in watermark.n3.run_wm_Lx_comp_sub_n3."""

from __future__ import annotations

import numpy as np
import torch

from .moss_io import decode_moss, encode_moss
from .plan import apply_plan, build_rare_cycle_plan
from .residual_feat import encode_hidden, layer_zq_from_ids
from .io import match_len


def embed_layer_tokens(model, wav: np.ndarray, table: dict, wm_layer: int, period: int = 3):
    tokens = encode_moss(model, wav)
    n_q, T = tokens.shape
    layers = [int(wm_layer)]
    plan = build_rare_cycle_plan(
        T,
        n_q=n_q,
        high_start=int(wm_layer),
        high_end=int(wm_layer),
        table=table,
        layers=layers,
        period=int(period),
    )
    mod, n_changed = apply_plan(tokens, plan)
    return tokens, mod, plan, int(n_changed)


def decode_codes(model, wav: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return match_len(wav, decode_moss(model, codes))


def _zq_full(q, ids_1t: torch.Tensor, max_time: int, mask: torch.Tensor) -> torch.Tensor:
    z_q = layer_zq_from_ids(q, ids_1t)
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
