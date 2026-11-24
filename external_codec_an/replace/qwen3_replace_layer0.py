import os
import torch
import random
import numpy as np
import soundfile as sf
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_qwen3_tts(device="npu:1"):
    tokenizer_path = r"/home/ma-user/work/yinian/pred_model/Qwen3-TTS-12Hz-1.7B-VoiceDesign/speech_tokenizer"
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_path, device_map=device, dtype=torch.bfloat16,
    )
    return tokenizer


def encode_Qwen3(tokenizer, audio_path):
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    encoded = tokenizer.encode(audio, sr=sr)
    code = encoded.audio_codes[0]          # (T, C)
    trans = code.transpose(0, 1)           # (C, T)
    return trans.long()                    # [num_quantizers, T]


def decode_Qwen3(tokenizer, tokens):
    """tokens: [num_quantizers, T]"""
    code = tokens.transpose(0, 1)          # (T, C)
    wavs, out_sr = tokenizer.decode({"audio_codes": [code]})
    return torch.from_numpy(wavs[0]).float(), out_sr


def perturb_one_layer(tokens, layer_idx, codebook_size=2048):
    """只随机替换指定的一层，其余层不变。"""
    perturbed = tokens.clone()
    T = perturbed.shape[1]
    perturbed[layer_idx, :] = torch.randint(
        0, codebook_size, (T,), device=tokens.device
    )
    return perturbed


def main():
    audio_dir = r"/home/ma-user/work/yinian/code/test/codec_an/audios/genres_original"
    output_dir = r"/home/ma-user/work/yinian/code/test/codec_an/audios/music/qwen3"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = load_qwen3_tts()
    n_layers = 16  # 层号 0..15，每轮只改一层

    for layer_idx in range(n_layers):
        save_dir = os.path.join(output_dir, f"repalce_{layer_idx}")
        os.makedirs(save_dir, exist_ok=True)
        for audio_name in os.listdir(audio_dir):
            audio_path = os.path.join(audio_dir, audio_name)
            audio_tokens = encode_Qwen3(tokenizer, audio_path)
            new_audio_tokens = perturb_one_layer(
                audio_tokens, layer_idx, codebook_size=2048
            )
            wav, sr = decode_Qwen3(tokenizer, new_audio_tokens)
            save_path = os.path.join(save_dir, audio_name)
            sf.write(save_path, wav, sr)


if __name__ == "__main__":
    main()
