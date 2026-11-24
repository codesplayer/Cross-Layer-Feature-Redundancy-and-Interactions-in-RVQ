"""Mel-Cepstral Distortion（pymcd 包；非其它代码仓库）。"""

from __future__ import annotations


class CalcuMCD:
    def __init__(self):
        from pymcd.mcd import Calculate_MCD

        self.mcd_tool = Calculate_MCD(MCD_mode="dtw")

    def mcd_file(self, ref_audio, rec_audio):
        return self.mcd_tool.calculate_mcd(ref_audio, rec_audio)
