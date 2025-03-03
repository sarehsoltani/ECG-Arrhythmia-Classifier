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

Start by installing the required Python libraries: pip install -r requirements.txt

   
### 2. Data Setup
Once the dependencies are installed, follow the steps below to prepare the data for training:

1. **Download the Dataset**: 
   - Download the **MIT-BIH Arrhythmia Dataset** from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?resource=download&select=mitbih_train.csv). 
   - Ensure the dataset is properly saved in the `/data/raw` directory of this repository.

2. **Data Preprocessing**: 
   - Before training, the dataset must be preprocessed. This is done in the `ECG_Heartbeat_Classification/preprocessing.ipynb`notebook.
   - Run the preprocessing notebook to clean and augment the ECG signals. The processed dataset will then be saved in the `/data/processed` directory, ready for model 
   input.
   The preprocessed data can also be accessed directly via this [Google Drive link](https://drive.google.com/drive/folders/1n1KG3qWTDousFy8LNICsIoTvNbXvemk_?usp=sharing).

### 3. **Model, Training, and Hyperparameter Tuning**:
   The `ECG_Heartbeat_Classification/baseline.ipynb`notebook contains the model architecture, training procedure, and hyperparameter tuning.
   This notebook performs the following tasks:

-  **Load the Preprocessed ECG Data**: 
   - The preprocessed ECG data, saved in the `/data/processed` directory, is loaded for training.

-  **Define the CNN Model**: 
   - A Convolutional Neural Network (CNN) model is defined to classify ECG signals into different heartbeat categories.

-  **Train the Model**: 
   - The model is trained using the preprocessed data, and various training strategies (such as dropout and batch normalization) are applied.

-  **Log Results with MLFlow**: 
   - During the training process, metrics (precision, recall, F1 score, accuracy, and AUC-ROC) and hyperparameters (learning rate and batch size) are logged to MLFlow for tracking and versioning.

-  **Optimize Hyperparameters using Optuna**: 
   - Optuna is used to perform hyperparameter optimization, tuning the model's learning rate and batch size to achieve better performance.

-  **Save the Trained Model**: 
   - The final trained model, after optimization and evaluation, is saved in the `/models` directory for future use and deployment.

### 4. Holdout Set Evaluation:
   Evaluate the model's performance on the holdout set by running the `ECG_Heartbeat_Classification/holdset.ipynb`notebook. This will load the best model, preprocess 
   the holdout set, and evaluate the model’s metrics.


