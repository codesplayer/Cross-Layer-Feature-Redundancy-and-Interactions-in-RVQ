# Cross-Layer Feature Redundancy and Interactions in RVQ

Code release for the paper *Cross-Layer Feature Redundancy and Interactions in RVQ*.

The package analyzes residual vector quantization (RVQ) in neural speech codecs through quantized-feature geometry, controlled layer editing, and re-encoding readback, then validates a training-free ABA watermark guided by those measurements. The reported watermark writes a structured key into an existing utterance and detects it in the re-encoded feature domain. No codec is retrained.

This directory is a self-contained snapshot for upload and reproduction. Shared library code lives in `base/`. Analysis and watermark experiments live in `other/` and `P1/`. A few paper scripts that still depend on the older `codec_an` tree are kept under `external_codec_an/` as reference copies.

---

## What the code does

1. **Geometry.** Signed energy decomposition, layer-pair cosine, and orthogonal-component cancellation on the actual RVQ path (Table II, Fig. 3).
2. **Structured ABA probes.** Single- and two-layer token writing, editing cost, and template separation \(\Delta_A\) (Fig. 4, Tables III–IV).
3. **Readback.** Aggregate representation cosine, same-layer feature cosine, and exact token agreement after an encode–decode round trip.
4. **Watermark.** ABA embedding on a chosen carrier (MOSS L9, key `[330, 372, 330]` in the paper) and blind feature-cosine detection, with attack, quality, and baseline scripts for Table V.

Random Replace One / Replace All (Fig. 2), Drop-1/Drop-2 (Fig. 5), the global-or-slide detector \(S_A \lor S_L\), and transition-count key construction are included as reference scripts under `external_codec_an/`. They are not guaranteed to run from this package alone. See [Paper map](#paper-map).

---

## Layout

```
paper_P1_RVQ/
  README.md                 this file
  PAPER_CODE_MAP.md         section-level script map
  main.py                   unified entry (smoke + experiment dispatch)
  data/manifest.jsonl       utterance index for mix_1500 and the FLEURS test split
  base/                     codecs, features, attacks, metrics, paths
  other/experiments/        geometry: energy, pair cosine, orthogonal cancellation
  P1/
    embed.py                ABA embedding
    detect.py               feature-cosine detection
    baselines.py            WavMark / AudioSeal wrappers
    experiments/            layer gap, two-layer probes, readback, attacks, quality
  external_codec_an/        reference copies; see that folder's README
    replace/                Fig. 2 random replacement
    drop/                   Fig. 5 Drop-1/2
    detect_cascade/         SA or SL cascade
    key_transition/         transition statistics for key selection
```

---

## Requirements

- Python 3.10+
- PyTorch with the device you intend to use (CUDA, Ascend NPU, or CPU)
- `transformers` (MOSS is loaded with `trust_remote_code`)
- `numpy`, `soundfile`, `librosa`

Paper-scale jobs also expect the codec weights and the utterance lists used in the manuscript (mix_1500 for development, 2040 FLEURS utterances for the held-out watermark test). Those files are not shipped here.

Checked locally: `audioLLM` conda env, MOSS on CUDA, one AISHELL clip. If a user-site PyTorch conflicts with the conda build, start Python with `PYTHONNOUSERSITE=1`. If `soundfile` cannot be installed into the conda env, install it into `paper_P1_RVQ/.deps`; `main.py` adds that directory to `sys.path` when it exists.

---

## Data

The utterances used in the paper are listed in `data/manifest.jsonl`. The first two lines describe the splits. Every later line is one clip.

| Split | Lines | Role in the paper | Contents |
|-------|------:|-------------------|----------|
| `mix_1500` | 1500 | Development: geometry, ABA probes, readback, carrier, key, and far01 thresholds | 500 AISHELL-1 (`.wav`) + 500 Emilia Chinese + 500 Emilia English (`.mp3`) |
| `fleurs_wm_n20` | 2040 | Held-out watermark test | FLEURS, 102 languages, 20 utterances each |

If the audio is already in this tree, each clip's `path` is relative to the package root (`data/mix_1500/...`, `data/fleurs_wm_n20/...`). If you publish the code without the waveforms, rebuild the same sets from `source` and `source_id`:

| `source` | How to fetch one clip |
|----------|------------------------|
| `AISHELL-1` | [OpenSLR SLR33](https://www.openslr.org/33/). `source_id` is the utterance id (`BAC009S0002W0133`). The file inside the tarball is `source_path` (`data_aishell/wav/S0002/BAC009S0002W0133.wav`). Apache-2.0; the AISHELL site limits use to academic research. |
| `Emilia` | [amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset). `source_id` is the Emilia id with the local `sub_` prefix removed (`EN_B00000_S00006_W000113` = language `EN`, batch `B00000`, speaker `S00006`, utterance `W000113`). CC BY-NC 4.0. Do not upload these clips to a public repository; ship this manifest and let readers download Emilia themselves. |
| `FLEURS` | [google/fleurs](https://huggingface.co/datasets/google/fleurs), config `hf_config` (for example `af_za`). Keep the example whose `id` equals `source_id`. CC BY 4.0; cite FLEURS when you redistribute the subset. |

```python
import json
from pathlib import Path

root = Path("paper_P1_RVQ")
for line in (root / "data" / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if row["record"] != "clip":
        continue
    local = root / row["path"]
    if local.is_file():
        wav = local
    else:
        wav = None  # resolve row["source"] + row["source_id"] from the upstream dataset
    print(row["split"], row["source"], row["source_id"], wav)
```

Point experiment `--wav-dir` arguments at `data/mix_1500` or `data/fleurs_wm_n20` when the files are present. The smoke test in the next section uses one short AISHELL file that is not required to be inside `mix_1500`.

## Paths

Defaults are resolved in `base/config.py` by taking the first path that exists on the machine. Typical Windows locations already listed there:

| Resource | Example path |
|----------|----------------|
| MOSS-Audio-Tokenizer | `e:\pred_model\MOSS\MOSS-Audio-Tokenizer` |
| Rare ABA table | `d:\data\codec_an\anas\watermark_rare_tokens\rare_aba.json` |
| Short smoke wavs | `e:\data\Clotho\aishell1\test\wav` |

DAC and Qwen paths are set inside the experiment scripts that load those codecs. Override them there, or pass the script’s own `--model` / path flags, before a multi-codec run.

The rare-token table used by the paper’s L9 key stores ABA `[330, 372, 330]` at layer 9.

---

## Quick start

From this directory:

```powershell
cd e:\code\water_marks\paper_P1_RVQ
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "$pwd;$pwd\.deps"
python main.py smoke --n-clips 1 --device cuda:0
```

`python main.py` with no subcommand runs the same smoke test. The report is written to `result_smoke/smoke_report.json`.

The smoke test does three things on one utterance:

1. Resolves the MOSS checkpoint, the rare ABA table, and a wav file, then loads the model.
2. Checks the RVQ energy identities on the encoded utterance (aggregate energy and the cross-term reconstruction).
3. Writes the layer-9 ABA pattern, scores the written feature against that token grid, and scores the same template after one encode–decode round trip on both the watermarked and the clean reconstruction.

A local CUDA run on `BAC009S0764W0121` (24 kHz, 32 codebooks, \(T=52\)) passed with:

| Check | Value |
|-------|--------|
| Energy identity error | \(\approx 1.1 \times 10^{-4}\) |
| Cross-term identity error | \(\approx 3 \times 10^{-5}\) |
| Direct template cosine | 0.669 |
| Re-encoded watermark / clean cosine | 0.537 / \(-0.134\) |
| Gap | 0.671 |
| L9 token agreement after re-encoding | 0.170 |

The last row matches the paper’s observation that most inserted tokens change under re-encoding while the feature template remains separable (reported token survival at MOSS L9 is 19.58%).

List every dispatcher command:

```powershell
python main.py list
```

---

## Minimal embedding and detection API

```python
from base.config import default_model_path, default_table_path
from base.moss_io import read_wav, encode_moss, decode_moss
from base.residual_feat import band_zq_of, codes_to_zq_sum, feat_cosine
from base.util import load_model
from P1.embed import embed_aba

model, sr = load_model(default_model_path(), "cuda:0")
wav = read_wav(r"e:\data\Clotho\aishell1\test\wav\BAC009S0764W0121.wav", sr)

emb = embed_aba(model, wav, layer=9, table_path=default_table_path())
template = codes_to_zq_sum(model, emb["mod"], 9, 9, layers=[9])

# Score the watermarked waveform before a further codec round trip.
feat = band_zq_of(model, emb["wav"], 9, 9)
print(feat_cosine(feat, template))

# Score after re-encoding, as in the paper’s detection setting.
wm = decode_moss(model, encode_moss(model, emb["wav"]))
print(feat_cosine(band_zq_of(model, wm, 9, 9), template))
```

`P1.detect.detect_aba` scores a waveform against a key template built by `P1.embed.key_template`. For a single utterance, build the template from the written code grid (`emb["mod"]`), as in the snippet above, so the template length matches the encoded frames.

---

## Larger experiments

`main.py` forwards arguments after `--` to the underlying module. Scale runs expect the mix_1500 manifest, codec weights, and usually multiple devices. Inspect `--help` on the target module before a full job.

```powershell
python main.py geometry -- --help
python main.py geometry -- --codecs moss --n-clips 3 --num-workers 1 --device cuda:0
python main.py layer-gap -- --codecs moss --n-clips 3 --num-workers 1 --device cuda:0
python main.py two-layer -- --codecs moss --n-clips 3 --num-workers 1 --device cuda:0
python main.py readback -- --codecs moss --n-clips 3 --num-workers 1 --device cuda:0
python main.py watermark-attack -- --help
python main.py watermark-quality -- --help
```

Equivalent module entry points:

| Command | Module |
|---------|--------|
| `geometry` | `other.experiments.run_rvq_af_n1500` |
| `layer-gap` | `P1.experiments.run_codecs_band_layer_gap` |
| `two-layer` | `P1.experiments.run_codecs_band_two_layer_random` |
| `readback` | `P1.experiments.run_codecs_multi_roundtrip_feat_n1500` |
| `watermark-attack` | `P1.experiments.run_anchor_attack_three_n1500` |
| `watermark-quality` | `P1.experiments.run_wm_clean_quality` |
| `fleurs-far01` | `P1.experiments.eval_fleurs_far01_from_anchor_clean` |

The paper’s detection protocol freezes far01 thresholds (target FAR \(\le 1\%\)) on mix_1500, then reports FAR, TPR, and AUC on 2040 paired FLEURS utterances with attack-matched unmarked negatives. Attack definitions and the frozen thresholds are part of those experiment scripts and the supplementary material.

---

## Paper map

| Paper item | Code |
|------------|------|
| Signed energy, layer-pair cosine, orthogonal cancellation (Table II, Fig. 3) | `other/experiments/run_rvq_af_n1500.py` |
| Random Replace One / Replace All (Fig. 2, Table I) | `external_codec_an/replace/` |
| Single-layer ABA and \(\Delta_A\) (Fig. 4, Table IV) | `P1/experiments/run_codecs_band_layer_gap.py` |
| Two-layer ABA interactions (Table III) | `P1/experiments/run_codecs_band_two_layer_random.py`, `other/experiments/_stat_layer_interaction_id.py` |
| Drop-1 / Drop-2 (Fig. 5) | `external_codec_an/drop/run_layer_pareto.py` |
| Aggregate, feature, and token readback | `P1/experiments/run_codecs_multi_roundtrip_feat_n1500.py` |
| ABA embed and feature-cosine detect | `P1/embed.py`, `P1/detect.py` |
| \(S_A \lor S_L\) and blind crop search | `external_codec_an/detect_cascade/_multi_channel.py` |
| Transition-guided key construction | `external_codec_an/key_transition/` |
| far01 attacks vs. WavMark and AudioSeal (Table V) | `P1/experiments/run_anchor_attack_three_n1500.py`, `P1/baselines.py`, `eval_fleurs_far01_from_anchor_clean.py` |
| Watermark quality | `P1/experiments/run_wm_clean_quality.py` |

`PAPER_CODE_MAP.md` marks each script as runnable in this package, partial, or reference-only.

---

## Scope

Included:

- The analysis and watermark code that corresponds to the paper’s main sections.
- Baseline wrappers used in the watermark comparison.
- Reference copies of scripts that have not been fully ported.

Left out on purpose:

- Later lines of work (conditional residual compensation, sign-flip watermarking) that the discussion lists as follow-up.
- Exploratory probes that do not appear in the paper.
- Model weights and stored experiment JSON or wav outputs.
- A public copy of the Emilia waveforms. `data/manifest.jsonl` is the index; readers download Emilia from the official release. AISHELL-1 and FLEURS clips can be recovered from the same file, and may also sit under `data/` in a local checkout.

`P1/channels.py` in this package is a stub. The cascade used for the paper’s blind detector is `external_codec_an/detect_cascade/_multi_channel.py`. Scripts in `external_codec_an/` were copied from the original tree and may still import packages that are not on this `PYTHONPATH`. Treat them as the source to port, or run them from the original tree. Details are in `external_codec_an/README.md`.

---

## Suggested reproduction order

1. `python main.py smoke --n-clips 1 --device cuda:0`
2. Geometry on a few clips: `python main.py geometry -- --codecs moss --n-clips 3 --num-workers 1 --device cuda:0`
3. Layer gap / \(\Delta_A\), two-layer probes, then readback, using the same small `--n-clips` before the full 1500.
4. Watermark attacks and quality once mix_1500 and the FLEURS split are mounted.
5. Fig. 2 and Fig. 5 from `external_codec_an/` after those scripts are ported or run in the original tree.
