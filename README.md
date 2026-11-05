\# Leakage-aware benchmarking of multimodal deep learning for autonomic activation recognition



Code and derived results for the paper (BSPC-D-26-12245).



\## Data

\- WESAD: https://doi.org/10.24432/C57K5T

\- CASE: https://doi.org/10.1038/s41597-019-0209-0

Raw data are not redistributed. Download them and run the preprocessing scripts.



\## Contents

\- code/      preprocessing, baselines, training and analysis code

\- results/   per-window predicted probabilities and explanation results used for all tables

\- logs/      data summaries produced by the preprocessing scripts



\## Reproduction

1\. pip install -r requirements.txt

2\. Run code/preprocess\_wesad.py and code/pre\_case.py

3\. Run the training and analysis code (seeds 0, 1, 2; leave-one-subject-out)



\## Environment

Python 3.13, PyTorch 2.11 (CUDA 13.0), see the paper (Section 3).



\## License

MIT

