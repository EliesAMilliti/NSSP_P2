# NSSP_P2 – EMG decoding mini-project

Code for **Neural Signals and Signal Processing – Mini-project 2**  
Variant: EMG-based hand movement decoding with NinaPro datasets.

The aim is to process surface EMG recordings to  
(i) classify hand gestures for a single subject,  
(ii) study generalisation across subjects, and  
(iii) regress finger joint angles for robotic hand control.

---

## Repository structure

- **`part1.ipynb` – Single-subject classification (NinaPro DB1, subject 2)**  
  - Load and visualise EMG data (S2_A1_E1).  
  - Pre-processing and sliding-window extraction.  
  - Feature extraction and normalisation.  
  - Random Forest classifier with validation and test evaluation.

- **`part2.ipynb` – Across-subject generalisation (NinaPro DB1)**  
  - Train/val/test splits across multiple subjects.  
  - Same feature family and classifier as Part 1.  
  - Experiments on how performance changes with the number of training subjects.

- **`part3.ipynb` – Regression of joint angles (NinaPro DB8, subject 1)**  
  - Load EMG and kinematics (file `S1_E1_A1.mat`).  
  - Time-series split into train/validation/test.  
  - Sliding-window EMG features (simple mean/var/max as the main baseline).  
  - Random Forest regressors (multi-output model and one model per joint).  
  - Evaluation with R² / MSE / MAE and analysis of performance stability across joints.  
  - Additional experiments with extended time- and frequency-domain EMG features are
    reported in the appendix section of the notebook.

- **`helpers.py`** – Utility functions shared by the notebooks  
  (data loading, windowing, feature extraction, metrics, plotting, etc.).

- **`best_model_part1.pkl`, `scaler_part1.pkl`** – Saved classifier and scaler
  used for Part 1.

---

## Data

This project uses the **NinaPro** datasets:

- **DB1** – used in Parts 1 and 2  
  Instructions: <https://ninapro.hevs.ch/instructions/DB1.html>
- **DB8** – used in Part 3  
  Instructions: <https://ninapro.hevs.ch/instructions/DB8.html>

Please download the required `.mat` files from the NinaPro website and place
them in a local `data/` folder (or update the paths at the top of each notebook).

---

## Dependencies and running the code

The code was tested with Python 3.x and the following main packages:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- `jupyter`

A minimal setup using `pip` could be:

```bash
pip install numpy scipy scikit-learn matplotlib seaborn pandas jupyter
```

Then start Jupyter and run the notebooks in order:
  part1.ipynb
  part2.ipynb
  part3.ipynb
Each notebook is organised so that all cells can be run top-to-bottom.
