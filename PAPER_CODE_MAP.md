# Paper section → code map

Paper: *Cross-Layer Feature Redundancy and Interactions in RVQ*

Status legend: **OK** = runnable under this package PYTHONPATH; **REF** = copied from `codec_an`, may need extra deps; **PARTIAL** = covers part of the claim.

---

## III-B Feature redundancy & interaction measurements

| Claim / figure | Script | Status |
|----------------|--------|--------|
| Signed energy E1/E2/E3, E±, EQr (Table II) | `other/experiments/run_rvq_af_n1500.py` (`experiment_a`) | OK |
| Layer-pair signed cosine Cij (Fig. 3) | same (`experiment_b`) | OK |
| Orthogonal cancellation C⊥ + temporal shuffle | same (`experiment_c`) | OK |
| Extra η / stats | `run_rvq_extra_eta_n1500.py`, `_stat_rvq_n1500.py` | OK |
| Random Replace One / Replace All (Fig. 2, Table I) | `external_codec_an/replace/*_replace_*.py` | REF |

## III-C ABA probes & controlled interventions

| Claim / figure | Script | Status |
|----------------|--------|--------|
| Single-layer ABA + ΔA (Fig. 4) | `P1/experiments/run_codecs_band_layer_gap.py` | OK |
| Two-layer ABA / interactions (Table III) | `run_codecs_band_two_layer_random.py`, `other/experiments/_stat_layer_interaction_id.py` | OK / OK |
| Drop-1 / Drop-2 (Fig. 5) | `external_codec_an/drop/run_layer_pareto.py` | REF |
| Drop-related WAV / attr | `P1/experiments/run_moss_aba_emotion_wav_n1500.py` (+ attr/ASR) | PARTIAL |

## III-D Readback & carrier evaluation

| Claim / figure | Script | Status |
|----------------|--------|--------|
| Aggregate / same-layer feature / token readback | `P1/experiments/run_codecs_multi_roundtrip_feat_n1500.py` | OK |
| Template separation ΔA vs cost (Table IV) | `run_codecs_band_layer_gap.py` | OK |

## III-E Watermark embedding & blind detection

| Claim / figure | Script | Status |
|----------------|--------|--------|
| ABA embed / key template | `P1/embed.py` | OK |
| Feature cosine detect | `P1/detect.py` | OK |
| SA ∨ SL cascade + blind crop search | `external_codec_an/detect_cascade/_multi_channel.py` | REF |
| Transition-guided key construction | `external_codec_an/key_transition/*` | REF |
| far01 / FLEURS attacks vs WavMark/AudioSeal (Table V) | `run_anchor_attack_three_n1500.py`, `eval_fleurs_far01_from_anchor_clean.py`, `P1/baselines.py` | OK |
| Quality vs baselines | `run_wm_clean_quality.py`, `run_band_wm_dual_quality.py` | OK |
| Key specificity / FAR01 ablations | `run_l9_true_vs_random_far01.py`, `run_l9_rare10_key_gap_matrix.py` | OK |

## Shared library (`base/`)

Used across analysis and watermark: `moss_io`, `codec_dac`, `codec_qwen`, `residual_feat`, `rare_tokens`, `detect_shared`, `channels`, `anchor_attacks`, `mild_attacks`, `lsd`, `mcd`, `quality`, `exp_parallel`, `npu_parallel`, `config`, …

---

## Suggested run order (reproduce narrative)

1. Geometry: `python -m other.experiments.run_rvq_af_n1500`
2. Layer gap / ΔA: `python -m P1.experiments.run_codecs_band_layer_gap`
3. Two-layer: `python -m P1.experiments.run_codecs_band_two_layer_random`
4. Readback: `python -m P1.experiments.run_codecs_multi_roundtrip_feat_n1500`
5. Watermark attacks: `python -m P1.experiments.run_anchor_attack_three_n1500`
6. (Optional REF) Replace / Drop / cascade from `external_codec_an/` with original `codec_an` PYTHONPATH
