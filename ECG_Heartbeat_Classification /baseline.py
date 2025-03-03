import numpy as np
import pandas as pd
import mlflow
import optuna
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalMaxPooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

mlflow.set_tracking_uri('mlruns')

# Load training and test data
x_train = pd.read_csv("data/processed/x_train.csv", header=None)
y_train = pd.read_csv("data/processed/y_train.csv", header=None)
x_test = pd.read_csv("data/processed/x_test.csv", header=None)
y_test = pd.read_csv("data/processed/y_test.csv", header=None)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# Convert to NumPy arrays and reshape for the CNN model
X_train = x_train.values[..., np.newaxis] 
Y_train = y_train.values.astype(np.int8)

X_test = x_test.values[..., np.newaxis]
Y_test = y_test.values.astype(np.int8)


def get_model(learning_rate, batch_size):
    nclass = 5  # Number of output classes
    inp = Input(shape=(187, 1))  # Input shape: (187 time-steps, 1 channel)

    # Convolutional layers
    x = Conv1D(16, kernel_size=5, activation='relu', padding='valid')(inp)
    x = Conv1D(16, kernel_size=5, activation='relu', padding='valid')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(rate=0.1)(x)

    x = Conv1D(32, kernel_size=3, activation='relu', padding='valid')(x)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='valid')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(rate=0.1)(x)

    x = Conv1D(32, kernel_size=3, activation='relu', padding='valid')(x)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='valid')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(rate=0.1)(x)

    x = Conv1D(256, kernel_size=3, activation='relu', padding='valid')(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='valid')(x)
    x = GlobalMaxPooling1D()(x)  # Global Max Pooling
    x = Dropout(rate=0.2)(x)

    # Fully connected layers
    x = Dense(64, activation='relu', name="dense_1")(x)
    x = Dense(64, activation='relu', name="dense_2")(x)
    x = Dense(nclass, activation='softmax', name="dense_3_mitbih")(x)

    # Model compilation
    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()  # Print the model summary
    return model

def plot_learning_curves(history):
    """
    Plot learning curves for training and validation accuracy/loss
    """
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()


def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    mlflow.set_experiment('ECG_Heartbeat_Classification_Experiment')
    
    # Start an MLFlow run for each trial
    with mlflow.start_run():
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('batch_size', batch_size)
        
        # Get the model
        model = get_model(learning_rate, batch_size)
        
        # Callbacks
        file_path = "/content/drive/MyDrive/ECG Classification/models/baseline_cnn_mitbih.h5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_accuracy", mode="max", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=3, verbose=2)
        callbacks_list = [checkpoint, early, redonplat]
        
        # Train the model
        history = model.fit(X_train, Y_train, epochs=100, validation_split=0.1, callbacks=callbacks_list, batch_size=batch_size, verbose=2)
        
        # Load the best model weights after training
        model.load_weights(file_path)
        
        # Plot the learning curves
        plot_learning_curves(history)
        
        # Predict the test data
        pred_test = model.predict(X_test)
        pred_test_class = np.argmax(pred_test, axis=-1)
        
        # Evaluate metrics
        precision = precision_score(Y_test, pred_test_class, average='macro')
        recall = recall_score(Y_test, pred_test_class, average='macro')
        f1 = f1_score(Y_test, pred_test_class, average='macro')
        accuracy = accuracy_score(Y_test, pred_test_class)
        roc_auc = roc_auc_score(Y_test, pred_test, multi_class='ovr', average='macro')
        
        # Log metrics to MLFlow
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('roc_auc', roc_auc)

        # Save the best model (after training, using the best weights)
        model.save("/content/drive/MyDrive/ECG Classification/models/baseline_cnn_mitbih_final_model.h5")  # Save the entire model

        # Log the model
        mlflow.keras.log_model(model, "model")

        # Print evaluation metrics
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC-ROC: {roc_auc:.4f}")

        # Return the f1 score for optimization
        return f1  # This will be used by Optuna to optimize the model

# Run the optimization process
study = optuna.create_study(direction='maximize')  # We want to maximize F1 score
study.optimize(objective, n_trials=5)  # Run 5 trials for hyperparameter tuning

# Print best hyperparameters and score
print(f"Best trial: {study.best_trial.value}")
print(f"Best trial hyperparameters: {study.best_trial.params}")
