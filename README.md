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

### 2. Data Setup


