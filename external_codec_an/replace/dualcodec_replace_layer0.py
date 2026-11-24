import os
import torch
import random
import numpy as np
import librosa
import soundfile as sf
import dualcodec

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

TARGET_SR = 24000
N_QUANTIZERS = 8
SEMANTIC_CODEBOOK_SIZE = 16384
ACOUSTIC_CODEBOOK_SIZE = 4096

# 离线权重目录（需事先用 huggingface-cli 下载）
DUALCODEC_CKPT_DIR = r"/home/ma-user/work/ynian/code/codec_an/model/dualcodec"
W2V_PATH = r"/home/ma-user/work/ynian/code/codec_an/model/w2v-bert-2.0"


def _load_wav(audio_path, target_sr=TARGET_SR):
    """librosa 读音频并直接 resample 到 24kHz。"""
    wav, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return wav.astype(np.float32)


def _move_inference_to_device(inference, device):
    """把 DualCodec 主模型与 w2v 语义塔迁到目标设备。"""
    inference.model.to(device)
    for key, obj in list(inference.semantic_cfg.items()):
        if isinstance(obj, (torch.nn.Module, torch.Tensor)):
            inference.semantic_cfg[key] = obj.to(device)
    inference.device = device


def load_dualcodec(
    model_id="12hz_v1",
    device="npu:4",
    dualcodec_model_path=DUALCODEC_CKPT_DIR,
    w2v_path=W2V_PATH,
):
    """离线加载 DualCodec。

    官方 Inference 只认 CUDA；无 CUDA 时先落到 CPU。
    若 device 为 npu:*，加载后再手动迁模，并关闭 autocast。
    """
    dualcodec_model = dualcodec.get_model(model_id, dualcodec_model_path)
    bootstrap = "cuda" if torch.cuda.is_available() else "cpu"
    inference = dualcodec.Inference(
        dualcodec_model=dualcodec_model,
        dualcodec_path=dualcodec_model_path,
        w2v_path=w2v_path,
        device=bootstrap,
        autocast=(bootstrap == "cuda"),
    )

    if str(device).startswith("npu"):
        _move_inference_to_device(inference, device)
        inference.autocast = False
    return inference


def encode_dualcodec(inference, audio_path, n_quantizers=N_QUANTIZERS):
    """编码为 [n_q, T] = cat(semantic[1], acoustic[n_q-1])。"""
    wav = _load_wav(audio_path, TARGET_SR)
    audio = torch.from_numpy(np.ascontiguousarray(wav)).float().view(1, 1, -1)
    with torch.no_grad():
        semantic_codes, acoustic_codes = inference.encode(
            audio, n_quantizers=n_quantizers
        )
    tokens = torch.cat([semantic_codes, acoustic_codes], dim=1)[0].long()
    return tokens


def decode_dualcodec(inference, tokens):
    """tokens: [n_q, T] → waveform [T]。"""
    codes = tokens.unsqueeze(0).to(inference.device)
    semantic_codes = codes[:, :1, :]
    acoustic_codes = codes[:, 1:, :]
    with torch.no_grad():
        out = inference.decode(semantic_codes, acoustic_codes)
    return out.squeeze()


def perturb_one_layer(tokens, layer_idx, codebook_size=ACOUSTIC_CODEBOOK_SIZE):
    """只随机替换 layer_idx 这一层，其余层不变。

    layer_idx 必须 >= 1（声学层）；第 0 层语义不替换。
    """
    if layer_idx < 1:
        raise ValueError("semantic layer 0 must not be replaced")
    perturbed = tokens.clone()
    T = perturbed.shape[1]
    perturbed[layer_idx, :] = torch.randint(
        0, codebook_size, (T,), device=tokens.device
    )
    return perturbed


def main():
    audio_dir = r"/home/ma-user/work/ynian/code/codec_an/audios/data/sub_audio"
    output_dir = r"/home/ma-user/work/ynian/code/codec_an/audios/speech/dualcodec/replace_one"
    os.makedirs(output_dir, exist_ok=True)

    inference = load_dualcodec()

    audio_names = [
        name for name in os.listdir(audio_dir)
        if os.path.isfile(os.path.join(audio_dir, name))
    ]

    for audio_name in audio_names:
        audio_path = os.path.join(audio_dir, audio_name)
        audio_tokens = encode_dualcodec(inference, audio_path)
        n = audio_tokens.size(0)

        for layer_idx in range(1, n):
            save_dir = os.path.join(output_dir, f"rp_one_{layer_idx}")
            os.makedirs(save_dir, exist_ok=True)
            new_audio_tokens = perturb_one_layer(audio_tokens, layer_idx)
            wav = decode_dualcodec(inference, new_audio_tokens)
            save_path = os.path.join(save_dir, audio_name)
            sf.write(save_path, wav.float().cpu().numpy(), TARGET_SR)


if __name__ == "__main__":
    main()
