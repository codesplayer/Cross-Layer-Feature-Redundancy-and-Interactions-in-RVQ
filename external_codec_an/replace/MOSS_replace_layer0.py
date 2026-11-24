import os
import torch
import random
import numpy as np
import soundfile as sf
import librosa
from transformers import AutoModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_DAC(device="npu:6"):
    model_path = r"/home/ma-user/work/ynian/code/codec_an/model/MOSS-Audio-Tokenizer"
    tokenizer = AutoModel.from_pretrained(model_path, device_map=device, trust_remote_code=True).eval()
    return tokenizer

def encode_MOSS(model, audio_path, device="npu:6"):
    wav, sr = librosa.load(audio_path, sr=model.sampling_rate)

    wav = torch.from_numpy(wav).float().unsqueeze(0)
    input_values = wav.unsqueeze(0).to(device)  # [1, 1, T]
    out = model.encode(input_values)
    codes = out.audio_codes                  # [num_quantizers, 1, T]
    return codes.squeeze(1).long()           # [num_quantizers, T]


def decode_MOSS(model, tokens):
    """tokens: [num_quantizers, T]"""
    codes = tokens.unsqueeze(1)              # [num_quantizers, 1, T]
    out = model.decode(codes)
    return out.audio.squeeze(0)              # [C, T]

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
    output_dir = r"/home/ma-user/work/ynian/code/codec_an/audios/speech/MOSS/replace_one"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = load_DAC()

    audio_names = [
        name for name in os.listdir(audio_dir)
        if os.path.isfile(os.path.join(audio_dir, name))
    ]

    for audio_name in audio_names:
        audio_path = os.path.join(audio_dir, audio_name)
        audio_tokens = encode_MOSS(tokenizer, audio_path)
        n = audio_tokens.size(0)  # 通常为 16

        # 每次只替换一层：layer 0, 1, ..., n-1
        for layer_idx in range(n):
            save_dir = os.path.join(output_dir, f"rp_one_{layer_idx}")
            os.makedirs(save_dir, exist_ok=True)
            new_audio_tokens = perturb_one_layer(audio_tokens, layer_idx)
            wav = decode_MOSS(tokenizer, new_audio_tokens)
            save_path = os.path.join(save_dir, audio_name)
            wav = wav.squeeze().float().cpu().numpy()
            sf.write(save_path, wav, tokenizer.sampling_rate)


if __name__ == '__main__':
    main()
