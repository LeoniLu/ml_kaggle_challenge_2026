# Project Replication Guide

This repository contains the code and resources required to reproduce the final machine learning model. 

For a high-level overview of our methodology, architecture, and findings, please refer to our project presentation:
👉 **[Download PDF Slide Deck](ML_Project_Presentation.pdf)**

---

## Model Replication Instructions - 3 Options

Reproducing the Final Selected **SVR** ModelThis repository provides three levels of reproducibility for the final Support Vector Regression (**SVR**) model using a sigmoid kernel and 3‑fold cross‑validation.

## Option (1) Shortcut (Direct Reproduction) Fastest way to reproduce the final model and generate the submission CSV.
```bash
  python repro_best_svr.py
```
This script load the model with the pre‑selected hyperparameters and will automatically generate the predicted **csv** file for submission.

## Option (2) Partial Linear Fine‑Tune (Within a Given Range)

```bash
  python main.py
```
The default main.py runs linear\_scann, which performs:

- Linear‑range hyperparameter search
    
- Up to 30 iterations, or early stopping if no improvement occurs between iterations

This produces a refined model based on the specified linear search range and generates submission **csv** file.

## Option (3) Full Hyperparameter Search

**Step 1 — Run Rough Exponential ScanModify main.py:**

- Comment out linear\_scann (line 75)
    
- Uncomment expo\_scann (line 71)

```bash
  python main.py
```

This generates a folder named output\_rough\_3fold, containing **json** files for each exponential degree with the best hyperparameters per iteration.

**Step 2 — Extract Best Rough‑Scan ResultsRun:**
```bash
  python temp\_scan\_scores.py
```

This script scans all **json** files in output\_rough\_3fold, identifies the best score, and prints the corresponding file path. Use this information to determine:

- min/max ranges for the linear scan
    
- \# of CV folds (if adjustments are needed)

**Step 3 — Run Final Linear Fine‑TuneModify main.py again:**

- Uncomment linear\_scann (line 75)
    
- Comment out expo\_scann (line 71)

Then run:
```bash
  python main.py
```

This produces the final fine‑tuned **SVR** model and the submission **csv**.


---
Archived Code & ResultsThe Archived Code and Archived Results directories contain:

- Earlier model attempts
    
- Hyperparameter exploration logs
    
- Intermediate experiments

Some files were overwritten during exploration, but most of the development history has been preserved for reference.

