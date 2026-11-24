"""音质 / 说话人相似度。"""

from __future__ import annotations

from .config import WAVLM_SV_PATH


class SpeakerSim:
    def __init__(self, device: str = "cuda:0", path: str = WAVLM_SV_PATH):
        from .speaker_sim import CalcuSpeakerSim

        self._sim = CalcuSpeakerSim(model_source=path, device=device)

    def sim_file(self, orig, test) -> float:
        return float(self._sim.sim_file(str(orig), str(test)))
