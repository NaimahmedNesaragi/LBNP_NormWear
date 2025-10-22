# LBNP NormWear Training Pipeline

This repository contains the complete data-preparation and training pipeline used for the **LBNP (Negative Pressure Lower-Body Experiment)** analysis project, based on the **NormWear** architecture.
It supports multimodal physiological data (e.g., LBNP, EKG, ABP, Photoplethysmography) and trains a dual-head model for both **classification** and **regression** tasks.

---

## 🧠 Project Overview

The aim of this project is to model physiological responses during graded lower-body negative-pressure (LBNP) sessions to estimate hemodynamic parameters such as **stroke volume (SV)** while also classifying session states or conditions.
This repository implements a two-phase transfer-learning strategy inspired by the NormWear paper:

1. **Phase 1 – Train Heads Only:**
   Freeze backbone and train classification + regression heads.
2. **Phase 2 – Fine-Tune Backbone:**
   Unfreeze the entire model for end-to-end optimization.

The code was developed under the **“Enhancing Prognostication and Mitigating Uncertainty” (2025–2028)** project funded by the Health South-East Regional Health Authority.

---

## 📁 Repository Structure

```
LBNP_NormWear/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── level7/
│   ├── data_prep.py        # Generates windowed .pkl samples and subject-wise folds
│   ├── train.py            # Two-phase training with mixed precision & TensorBoard logging
│   ├── model.py            # NormWear model definition
│   ├── dataset.py          # Dataset loader & augmentations
│   ├── utils.py            # Metrics, loss functions, helpers
│   ├── folds.json          # Final subject-wise splits (train/test per fold)
│   └── ...
│
├── results/
│   ├── checkpoints/
│   ├── tensorboard_logs/
│   └── training_logs.csv
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

## ⚙️ Environment Setup

```bash
conda create -n normwear_gpu python=3.10
conda activate normwear_gpu
pip install -r requirements.txt
```

---

## 🧾 Data

The preprocessed data consists of windowed `.pkl` files and corresponding `train_test_split.json` metadata for each fold.
Each sample contains synchronized physiological channels such as:

* `LBNP`
* `EKG_SD50`
* `Photopleth`
* `ABP_rd_ledning`
* `Nexfin_ABP`

with **`Nexfin_SV`** used as the regression label.

Data is not included in this repository but should be organized as:

```
data/
 ├── Subject_1/
 ├── Subject_2/
 └── ...
```

---

## 🚀 Training

To train on a specific fold:

```bash
python train.py --fold 1
```

Available arguments:

| Argument            | Description            | Default           |
| ------------------- | ---------------------- | ----------------- |
| `--fold`            | Fold number (1–3)      | 1                 |
| `--epochs`          | Total epochs per phase | Defined in script |
| `--mixed_precision` | Enable AMP training    | True              |

Training proceeds in two automatic phases:

1. **Phase 1:** Train classification + regression heads only
2. **Phase 2:** Fine-tune full model

---

## 📊 Logging and Outputs

* **TensorBoard:** Training and validation metrics logged under `results/tensorboard_logs/`
* **CSV Logs:** Per-epoch metrics stored in `results/training_logs.csv`
* **Checkpoints:**

  * `best_heads_foldX.pth` after Phase 1
  * `best_full_foldX.pth` after Phase 2

---

## 🧩 Key Metrics

* **Classification:** Accuracy / F1-score
* **Regression:** RMSE (unscaled), MAE
* **Loss Function:** Weighted combination of classification + regression loss

---

## 🧑‍🤝‍🧑 Collaboration

To contribute:

```bash
git clone https://github.com/<yourname>/LBNP_NormWear.git
git checkout -b feature/<new_feature_name>
# Make changes, commit, and open a pull request
```

---

## 📚 Citation & Acknowledgement

If you use this repository, please cite the following work:

> Nesaragi N. et al. (2025). *Enhancing Prognostication and Mitigating Uncertainty: Improving Reliability of EEG Assessments in Coma after Cardiac Arrest.*
> Health South-East Regional Health Authority, Norway (2025–2028).

Also acknowledge the base model:

> Vu T.H. et al. (2023). *NormWear: Normalizing Wearable Physiology for Cross-Subject Robustness.*

---

## 📬 Contact

For collaboration or technical questions:
**Naim Ahmed Nesaragi** — *Oslo, Norway*
📧 [naimahmed@example.com](mailto:naimahmed@example.com)  |  GitHub [@your-username](https://github.com/your-username)
