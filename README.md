<div align="center">  

# The NASA and DNV Challenge on Optimization under Uncertainty 
## Uncertainty-Aware Optimization in Engineered Systems via Gradient Boosting and Differential Evolution

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**[Dibakar Roy Sarkar](https://scholar.google.com/citations?user=Sz4nHdYAAAAJ&hl=en&oi=ao), [Sukanta Basu](https://scholar.google.com/citations?hl=en&user=08bv9p8AAAAJ), [Lance Manuel](https://scholar.google.com/citations?hl=en&user=NvlDB08AAAAJ), [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en)**

</div>

## Overview

This repository contains the machine learning codes developed to address the [NASA-DNV Challenge on Optimization under Uncertainty 2025](UQ_Challenge.pdf). The challenge, jointly posed by NASA and DNV, involves uncertainty quantification and optimization tasks for engineered systems.

Our work was presented at the UQ Challenge Sessions of the **35th European Safety and Reliability (ESREL) Conference** held in Norway. 

📄 **Paper**: [ESREL-SRA-E2025-P7698](https://rpsonline.com.sg/proceedings/esrel-sra-e2025/html/ESREL-SRA-E2025-P7698.html) (also available as [ESREL-SRA-E2025-P7698.pdf](ESREL-SRA-E2025-P7698.pdf) in this repository)  
🎥 **Presentation**: [Recorded talk](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/Eo6XL-LpD6ZOi9YAEh4gbqYBnehYqBaV6ees0N68VL0sPA?e=hIcXBC)  
📊 **Scoring**: [UQ_challenge_scoring.pdf](UQ_challenge_scoring.pdf)

## Repository Structure

```
2025_NASA_DNV_UQ/
├── Problem_1_1/                  # Sub-problem 1.1: Surrogate-based UQ
│   ├── 01_Preprocess.py          # Feature extraction from response time series
│   ├── 02_Training.py            # AutoML training (gradient boosting via FLAML)
│   ├── 03_Testing.py             # Model evaluation and prediction
│   └── output_*.log              # Training logs for various configurations
│
├── Problem_1_3/                  # Sub-problem 1.3: Prediction intervals
│   ├── Bound_Q1_c_res.py         # Bound estimation and prediction intervals
│   ├── job_submit.sh             # SLURM job submission script
│   └── prediction_intervals_results/  # Output plots and interval data
│
├── Problem_2/                    # Problem 2: Optimization under uncertainty
│   ├── p2_q1/DE1/                # Question 1 — Differential Evolution
│   ├── p2_q2/DE1/                # Question 2 — Differential Evolution
│   └── p2_q3/DE1/                # Question 3 — Differential Evolution
│
├── UQ_Challenge.pdf              # Official challenge description
├── UQ_challenge_scoring.pdf      # Scoring results from NASA and DNV
├── ESREL-SRA-E2025-P7698.pdf     # Published conference paper
├── CITATION.cff                  # Citation metadata
├── CONTRIBUTING.md               # Contribution guidelines
└── LICENSE                       # GPL-3.0 License
```

## Installation

### Prerequisites

- Python 3.8 or higher

### Install Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels antropy nolds flaml
```

## Usage

### Problem 1.1 — Surrogate-Based UQ

Run the scripts sequentially:

```bash
# Step 1: Extract statistical features from response time series
python Problem_1_1/01_Preprocess.py

# Step 2: Train gradient boosting surrogate models using FLAML AutoML
python Problem_1_1/02_Training.py

# Step 3: Evaluate trained models on test data
python Problem_1_1/03_Testing.py
```

### Problem 1.3 — Prediction Intervals

```bash
python Problem_1_3/Bound_Q1_c_res.py
```

### Problem 2 — Optimization Under Uncertainty

Each sub-question uses Differential Evolution for optimization:

```bash
python Problem_2/p2_q1/DE1/P2_q1_DE1.py
python Problem_2/p2_q2/DE1/P2_q2_DE1.py
python Problem_2/p2_q3/DE1/P2_q3_DE1.py
```

> **Note**: The scripts contain hardcoded paths for data directories that may need to be updated for your system. The SLURM job submission scripts (`job_submit.sh`) are provided for HPC cluster execution.

## Results

The scoring results published by NASA and DNV are available in [UQ_challenge_scoring.pdf](UQ_challenge_scoring.pdf). Prediction interval visualizations and probability curves can be found in the `Problem_1_3/prediction_intervals_results/` directory.

## Citation

If you find this work useful, please cite us:

```bibtex
@inproceedings{roy2025uncertainty,
  title={Uncertainty-Aware Optimization in Engineered Systems via Gradient Boosting and Differential Evolution},
  author={Roy Sarkar, Dibakar and Basu, Sukanta and Manuel, Lance and Goswami, Somdatta},
  booktitle={Proceedings of the 35th European Safety and Reliability & the 33rd Society for Risk Analysis Europe Conference},
  year={2025},
  publisher={Research Publishing, Singapore},
  doi={10.3850/978-981-94-3281-3_ESREL-SRA-E2025-P7698-cd}
}
```

You can also use the [CITATION.cff](CITATION.cff) file for automated citation tools.

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or further information, feel free to contact Dibakar Roy Sarkar at droysar1[@]jh[.]edu.
