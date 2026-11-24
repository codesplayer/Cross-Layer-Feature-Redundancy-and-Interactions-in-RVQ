import os
import torch
import random
import numpy as np
import soundfile as sf
from speechtokenizer import SpeechTokenizer

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEFAULT_CODEBOOK_SIZE = 1024


def _load_wav(audio_path, target_sr):
    """soundfile 读音频并重采样，避免 torchaudio/torchcodec。"""
    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if int(sr) != int(target_sr):
        n_new = int(round(len(wav) * float(target_sr) / float(sr)))
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
    return wav


def load_ST(device="npu:5"):
    dir_path = "/home/ma-user/work/ynian/code/codec_an/model/SpeechTokenizer/speechtokenizer_hubert_avg"
    config_path = os.path.join(dir_path, "config.json")
    ckpt_path = os.path.join(dir_path, "SpeechTokenizer.pt")
    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model.to(device).eval()
    return model


def encode_ST(model, audio_path):
    """编码为 [n_q, T]。SpeechTokenizer.encode 输出 (n_q, B, T)。"""
    device = next(model.parameters()).device
    wav = _load_wav(audio_path, model.sample_rate)
    wav = torch.from_numpy(np.ascontiguousarray(wav)).float().view(1, 1, -1).to(device)
    with torch.no_grad():
        codes = model.encode(wav)          # (n_q, B, T)
    return codes.squeeze(1).long()         # [n_q, T]


def decode_ST(model, tokens):
    """tokens: [n_q, T] → waveform [T]。decode 输入 (n_q, B, T)。"""
    codes = tokens.unsqueeze(1)            # (n_q, 1, T)
    with torch.no_grad():
        out = model.decode(codes)          # (B, 1, T)
    return out.squeeze()                   # [T]


def perturb_one_layer(tokens, layer_idx, codebook_size=DEFAULT_CODEBOOK_SIZE):
    """只随机替换 layer_idx 这一层，其余层不变。"""
    perturbed = tokens.clone()
    T = perturbed.shape[1]
    perturbed[layer_idx, :] = torch.randint(
        0, codebook_size, (T,), device=tokens.device
    )
    return perturbed


def main():
    audio_dir = r"/home/ma-user/work/ynian/code/codec_an/audios/data/sub_audio"
    output_dir = r"/home/ma-user/work/ynian/code/codec_an/audios/speech/ST/replace_one"
    os.makedirs(output_dir, exist_ok=True)

    model = load_ST()
    sr = int(model.sample_rate)

    audio_names = [
        name for name in os.listdir(audio_dir)
        if os.path.isfile(os.path.join(audio_dir, name))
    ]

    for audio_name in audio_names:
        audio_path = os.path.join(audio_dir, audio_name)
        audio_tokens = encode_ST(model, audio_path)
        n = audio_tokens.size(0)

        # 每次只替换一层：layer 0, 1, ..., n-1
        for layer_idx in range(n):
            save_dir = os.path.join(output_dir, f"rp_one_{layer_idx}")
            os.makedirs(save_dir, exist_ok=True)
            new_audio_tokens = perturb_one_layer(audio_tokens, layer_idx)
            wav = decode_ST(model, new_audio_tokens)
            save_path = os.path.join(save_dir, audio_name)
            sf.write(save_path, wav.float().cpu().numpy(), sr)


if __name__ == "__main__":
    main()
