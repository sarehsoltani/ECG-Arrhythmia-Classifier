# ECG Heartbeat Classification Using Deep Neural Networks

## Overview

This project aims to perform **ECG heartbeat classification** leveraging **deep neural network architectures**. The focus is on developing a robust classification model that can accurately identify various heartbeat types, including normal and arrhythmic patterns, from ECG signals. The project explores advanced techniques such as **transfer learning** to improve the model's performance.

The dataset used is the **MIT-BIH Arrhythmia Dataset**, sourced from **Physionet**, which consists of **109,446 samples** categorized into five classes. The ECG signals have a sampling frequency of **125 Hz**, meaning each signal has 125 samples per second. This dataset has been preprocessed and segmented, with each segment representing a single heartbeat.

### Dataset Classes:
- **0 (N)**: Normal beat
- **1 (S)**: Supraventricular premature beat
- **2 (V)**: Premature ventricular contraction
- **3 (F)**: Fusion of ventricular and normal beat
- **4 (Q)**: Unclassifiable beat

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?resource=download&select=mitbih_train.csv).

## Running the Code

To replicate the experiments and run the code, follow the instructions below:

### 1. Install Dependencies

Start by installing the required Python libraries:

```bash
pip install -r requirements.txt


```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ECG_Heartbeat_Classification and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ECG_Heartbeat_Classification   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ECG_Heartbeat_Classification a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

