# External copies from `e:\code\codec_an`

These scripts support paper figures that are **not yet fully migrated** into the standalone `water_marks` package.

| Subdir | Paper use | Notes |
|--------|-----------|-------|
| `replace/` | Fig. 2 / Table I random layer replace | Historically under `codec_an/codecs/experiments/` |
| `drop/` | Fig. 5 Drop-1/2 | `run_layer_pareto.py` |
| `detect_cascade/` | Eq. (25) SA ∨ SL | `_multi_channel.py`; `P1/channels.py` in this package is a stub |
| `key_transition/` | §III-E.1 transition counts for key selection | Analysis utilities |

**Do not assume** `PYTHONPATH=paper_P1_RVQ` alone runs these. Prefer documenting results as precomputed artifacts, or re-run with the original `codec_an` tree until they are ported.
