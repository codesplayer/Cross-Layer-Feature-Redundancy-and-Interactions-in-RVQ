import torch
import random
import os
import numpy as np
import soundfile as sf
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from model import DAC
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_DAC(device="npu:5"):
    model_path = r"/home/ma-user/work/ynian/code/codec_an/model/descript-audio-codec-24khz"
    tokenizer = DAC.from_pretrained(model_path)
    tokenizer.to(device).eval()
    return tokenizer


def encode_DAC(model, audio_path):
    """DAC.encode 接受文件路径，返回 (zq, s)"""
    _, s = model.encode(audio_path)        # s: [1, N, T]
    return s.squeeze(0).long()


def decode_DAC(model, tokens):
    """tokens: [N, T]"""
    s = tokens.unsqueeze(0)               # [1, N, T]
    out = model.decode(s=s)
    return out.squeeze(0)

def perturb_one_layer(tokens, layer_idx, codebook_size=1024):
    """只随机替换 layer_idx 这一层，其余层不变。"""
    perturbed = tokens.clone()
    T = perturbed.shape[1]
    perturbed[layer_idx, :] = torch.randint(
        0, codebook_size, (T,), device=tokens.device
    )
    return perturbed


def main():
    audio_dir = r"/home/ma-user/work/ynian/code/codec_an/audios/data/sub_audio"
    output_dir = r"/home/ma-user/work/ynian/code/codec_an/audios/speech/dac/replace_one"
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = load_DAC()

    audio_names = [
        name for name in os.listdir(audio_dir)
        if os.path.isfile(os.path.join(audio_dir, name))
    ]

    for audio_name in audio_names:
        audio_path = os.path.join(audio_dir, audio_name)
        audio_tokens = encode_DAC(tokenizer, audio_path)
        n = audio_tokens.size(0)  # 通常为 16

        # 每次只替换一层：layer 0, 1, ..., n-1
        for layer_idx in range(n):
            save_dir = os.path.join(output_dir, f"rp_one_{layer_idx}")
            os.makedirs(save_dir, exist_ok=True)
            new_audio_tokens = perturb_one_layer(audio_tokens, layer_idx)
            wav = decode_DAC(tokenizer, new_audio_tokens)
            save_path = os.path.join(save_dir, audio_name)
            wav = wav.squeeze().float().cpu().numpy()
            sf.write(save_path, wav, 24000)


if __name__ == "__main__":
    main()
