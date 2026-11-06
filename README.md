# Leakage-aware, subject-independent benchmarking of multimodal deep learning for autonomic activation recognition

Code and derived results for the accompanying manuscript (BSPC-D-26-12245).
The study evaluates a lightweight CNN-GRU-attention network (Hybrid, 19,491 parameters) against deep and classical
baselines on WESAD (15 subjects) and CASE (30 subjects) under one leave-one-subject-out protocol.

## Data (not redistributed)

* WESAD: https://doi.org/10.24432/C57K5T (scientific, non-commercial use, credit to the data set owners)
* CASE: https://doi.org/10.1038/s41597-019-0209-0
Derived window arrays (wesad\_windows.npz, case\_windows.npz): to be archived with a DOI.

## Repository layout

|Path|Content|
|-|-|
|`code/preprocess\_wesad.py`, `code/pre\_case.py`|Filtering, per-subject z-scoring, windowing and resampling; produce `wesad\_windows.npz` and `case\_windows.npz` (see the header of each script for input and output paths)|
|`code/ml\_baseline.py`|SVM and Random Forest on 20 handcrafted features|
|`code/common.py`|Hybrid network, CNN-only and RNN-only baselines, augmentation, focal loss, LOSO fold runner|
|`code/models\_extra.py`|Transformer, cross-modal attention and fusion variants|
|`code/wesad\_benchmark.py`|WESAD LOSO benchmark, modality, attention and recurrence ablations, fusion rules|
|`code/case\_validation.py`|CASE: LOSO within CASE and zero-shot transfer from WESAD|
|`code/xai\_results.py`|Robustness, attention entropy, R-peak alignment, Integrated Gradients, deletion/insertion|
|`code/xai\_stability.py`|Stability of Integrated Gradients maps across 45 independently trained models|
|`code/analysis.py`|Statistics behind the tables (bootstrap CI, Wilcoxon, Holm, Friedman, oracle-offset analysis)|
|`code/verify.py`|Integrity check of the repository and of the key numbers|
|`results/`|Saved per-window predicted probabilities and explanation results; keys are `(model, seed, subject)`|
|`logs/`|Data summaries written by the preprocessing scripts|

## Result files

|File|Produced by|Content|
|-|-|-|
|`results/results.pkl`|`wesad\_benchmark.py`|15 configurations x 3 seeds x 15 subjects (Hybrid, modality subsets, deep baselines, ablations)|
|`results/results\_ml.pkl`|`ml\_baseline.py`|SVM (seed 0) and Random Forest (3 seeds)|
|`results/results\_fusion.pkl`|`wesad\_benchmark.py --group fusion`|Three fusion rules x 3 seeds x 15 subjects|
|`results/results\_case.pkl`|`case\_validation.py`|CASE LOSO and zero-shot, 3 models x 3 seeds x 30 subjects|
|`results/xai\_results.pkl`|`xai\_results.py`|Robustness, attention and attribution results for 15 subjects|
|`results/xai\_stability.pkl`|`xai\_stability.py`|45 attribution-map sets (15 subjects x 3 seeds)|

## Reproduction

1. Install: `pip install -r requirements.txt` (GPU scripts need PyTorch with CUDA; analysis needs only numpy, scipy, scikit-learn and pandas).
2. Recompute every table from the saved results (CPU, no GPU needed): `python code/analysis.py --results results`
3. Check the repository and the key numbers: `python code/verify.py`
4. Retrain from the windows (GPU, several hours; each script saves after every fold and resumes if interrupted):

```
python code/wesad\_benchmark.py --data DATA\_DIR --out OUT\_DIR --group all
python code/case\_validation.py --data DATA\_DIR --out OUT\_DIR
python code/xai\_results.py     --data DATA\_DIR --out OUT\_DIR
python code/xai\_stability.py   --data DATA\_DIR --out OUT\_DIR
```

Add `--smoke` to any script for a one-subject, one-epoch test run (all GPU scripts were smoke-tested).

## Notes on reproducibility

* Seeds 0, 1, 2; subject splits are deterministic (Section 3 of the paper). GPU kernels are not bit-wise deterministic for
GRU and attention layers, so retraining the Hybrid reproduces results only approximately.
* For the CNN-only and RNN-only models, retraining subject S2 with seed 0 on a different platform reproduced the saved
probabilities exactly (maximum difference 0.0).
* The experiments were run on Kaggle and on Google Colab (NVIDIA Tesla T4, Python 3.13, PyTorch 2.11 with CUDA 13.0).
* The loops for the Hybrid and the modality subsets in `wesad\_benchmark.py` were reassembled from the original notebooks;
the other models, ablations and fusion rules are taken from the notebooks unchanged.

## License

MIT (see `LICENSE`).

