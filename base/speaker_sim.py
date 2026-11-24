import os
import json
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

# WavLM speaker verification；本地目录或 HuggingFace ID
MODEL_SOURCE = r"/data1/home/ynian/pred_models/wavlm-base-plus-sv"
# 若本地没有，可改为: "microsoft/wavlm-base-plus-sv"
TARGET_SR = 16000


def _resolve_device(device):
    if device is not None:
        return device
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu:7"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class CalcuSpeakerSim:
    """用 WavLM-Base-Plus-SV 提取说话人 embedding，计算重构与原始语音的余弦相似度。"""

    def __init__(
        self,
        model_source=MODEL_SOURCE,
        sr=TARGET_SR,
        device=None,
    ):
        """
        Args:
            model_source: HuggingFace ID 或本地模型目录
            sr: 目标采样率（WavLM-SV 使用 16 kHz）
            device: 推理设备，如 "cpu"、"cuda:0"、"npu:4"；None 时自动选择
        """
        self.model_source = model_source
        self.sr = sr
        self.device = _resolve_device(device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_source)
        self.model = WavLMForXVector.from_pretrained(model_source)
        self.model.to(self.device).eval()
        self._cosine = torch.nn.CosineSimilarity(dim=-1)

    def _load_mono(self, signal):
        if isinstance(signal, str):
            y, _ = librosa.load(signal, sr=self.sr, mono=True)
            return y.astype(np.float32)
        return np.asarray(signal, dtype=np.float32).squeeze()

    def extract_embeddings(self, audios):
        """批量提取 L2 归一化的 speaker embeddings。

        Args:
            audios: 音频路径或 waveform 列表

        Returns:
            torch.Tensor: [N, D]，已 normalize
        """
        waves = [self._load_mono(a) for a in audios]
        inputs = self.feature_extractor(
            waves,
            sampling_rate=self.sr,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings
        return F.normalize(embeddings, dim=-1)

    def compute(self, ref, deg):
        """
        计算说话人余弦相似度。

        Args:
            ref: 参考信号 (numpy array) 或音频文件路径
            deg: 重建信号 (numpy array) 或音频文件路径

        Returns:
            float: cosine similarity
        """
        embeddings = self.extract_embeddings([ref, deg])
        similarity = self._cosine(embeddings[0], embeddings[1])
        return float(similarity.detach().cpu())

    def sim_file(self, ref_path, deg_path):
        """从音频文件计算说话人相似度。"""
        return self.compute(ref_path, deg_path)


def main():
    ref_dir = "/home/ma-user/work/ynian/code/codec_an/audios/data/sub_audio"
    rec_dirs = "/home/ma-user/work/ynian/code/codec_an/audios/speech"
    result_dirs = "/home/ma-user/work/ynian/code/codec_an/audios/result/speaker_sim"

    calcu = CalcuSpeakerSim(device="npu:7")
    os.makedirs(result_dirs, exist_ok=True)

    for model_name in os.listdir(rec_dirs):
        model_dir = os.path.join(rec_dirs, model_name)
        if not os.path.isdir(model_dir):
            continue
        re_model = os.path.join(result_dirs, model_name)
        for re_type in os.listdir(model_dir):
            model_re = os.path.join(model_dir, re_type)
            if not os.path.isdir(model_re):
                continue
            save_rep = os.path.join(re_model, re_type)
            os.makedirs(save_rep, exist_ok=True)
            for num_layer in os.listdir(model_re):
                layer_dir = os.path.join(model_re, num_layer)
                if not os.path.isdir(layer_dir):
                    continue
                json_path = os.path.join(save_rep, f"{model_name}_{num_layer}.json")
                with open(json_path, "w", encoding="utf-8") as writer:
                    for audio_name in os.listdir(layer_dir):
                        ref_audio = os.path.join(ref_dir, audio_name)
                        audio_path = os.path.join(layer_dir, audio_name)
                        if not os.path.isfile(audio_path):
                            continue
                        sim = calcu.sim_file(ref_audio, audio_path)
                        simi_dict = {
                            "audio": audio_name,
                            "speaker_sim": round(sim, 4),
                            "rec_audio": audio_path,
                        }
                        writer.write(f"{json.dumps(simi_dict, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
