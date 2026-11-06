# Leakage-aware, subject-independent benchmarking of multimodal deep learning for autonomic activation recognition

Code and derived results for the manuscript BSPC-D-26-12245 [update with the final title and citation].
The study evaluates a lightweight CNN-GRU-attention network (Hybrid, 19,491 parameters) against deep and classical
baselines on WESAD (15 subjects) and CASE (30 subjects) under one leave-one-subject-out protocol.

## Data (not redistributed)
- WESAD: https://doi.org/10.24432/C57K5T (scientific, non-commercial use, credit to the data set owners)
- CASE: https://doi.org/10.1038/s41597-019-0209-0
Derived window arrays (`wesad_windows.npz`, `case_windows.npz`): [Zenodo DOI to be added].

## Repository layout
| Path | Content |
|---|---|
| `code/preprocess_wesad.py`, `code/pre_case.py` | Filtering, per-subject z-scoring, windowing and resampling; produce `wesad_windows.npz` and `case_windows.npz` (see the header of each script for input and output paths) |
| `code/ml_baseline.py` | SVM and Random Forest on 20 handcrafted features |
| `code/common.py` | Hybrid network, CNN-only and RNN-only baselines, augmentation, focal loss, LOSO fold runner |
| `code/models_extra.py` | Transformer, cross-modal attention and fusion variants |
| `code/wesad_benchmark.py` | WESAD LOSO benchmark, modality, attention and recurrence ablations, fusion rules |
| `code/case_validation.py` | CASE: LOSO within CASE and zero-shot transfer from WESAD |
| `code/xai_results.py` | Robustness, attention entropy, R-peak alignment, Integrated Gradients, deletion/insertion |
| `code/xai_stability.py` | Stability of Integrated Gradients maps across 45 independently trained models |
| `code/analysis.py` | Statistics behind the tables (bootstrap CI, Wilcoxon, Holm, Friedman, oracle-offset analysis) |
| `code/verify.py` | Integrity check of the repository and of the key numbers |
| `results/` | Saved per-window predicted probabilities and explanation results; keys are `(model, seed, subject)` |
| `logs/` | Data summaries written by the preprocessing scripts |

## Result files
| File | Produced by | Content |
|---|---|---|
| `results/results.pkl` | `wesad_benchmark.py` | 15 configurations x 3 seeds x 15 subjects (Hybrid, modality subsets, deep baselines, ablations) |
| `results/results_ml.pkl` | `ml_baseline.py` | SVM (seed 0) and Random Forest (3 seeds) |
| `results/results_fusion.pkl` | `wesad_benchmark.py --group fusion` | Three fusion rules x 3 seeds x 15 subjects |
| `results/results_case.pkl` | `case_validation.py` | CASE LOSO and zero-shot, 3 models x 3 seeds x 30 subjects |
| `results/xai_results.pkl` | `xai_results.py` | Robustness, attention and attribution results for 15 subjects |
| `results/xai_stability.pkl` | `xai_stability.py` | 45 attribution-map sets (15 subjects x 3 seeds) |

## Reproduction
1. Install: `pip install -r requirements.txt` (GPU scripts need PyTorch with CUDA; analysis needs only numpy, scipy, scikit-learn and pandas).
2. Recompute every table from the saved results (CPU, no GPU needed): `python code/analysis.py --results results`
3. Check the repository and the key numbers: `python code/verify.py`
4. Retrain from the windows (GPU, several hours; each script saves after every fold and resumes if interrupted):
```
python code/wesad_benchmark.py --data DATA_DIR --out OUT_DIR --group all
python code/case_validation.py --data DATA_DIR --out OUT_DIR
python code/xai_results.py     --data DATA_DIR --out OUT_DIR
python code/xai_stability.py   --data DATA_DIR --out OUT_DIR
```
Add `--smoke` to any script for a one-subject, one-epoch test run (all GPU scripts were smoke-tested).

## Notes on reproducibility
- Seeds 0, 1, 2; subject splits are deterministic (Section 3 of the paper). GPU kernels are not bit-wise deterministic for
  GRU and attention layers, so retraining the Hybrid reproduces results only approximately.
- For the CNN-only and RNN-only models, retraining subject S2 with seed 0 on a different platform reproduced the saved
  probabilities exactly (maximum difference 0.0).
- The experiments were run on Kaggle and on Google Colab (NVIDIA Tesla T4, Python 3.13, PyTorch 2.11 with CUDA 13.0).
- The loops for the Hybrid and the modality subsets in `wesad_benchmark.py` were reassembled from the original notebooks;
  the other models, ablations and fusion rules are taken from the notebooks unchanged.
- The fusion experiment was run twice. `results/results_fusion.pkl` is the first run; [the repeated run is in
  `results/results_fusion_run2.pkl` / add or delete this line].

## License
MIT (see `LICENSE`).
