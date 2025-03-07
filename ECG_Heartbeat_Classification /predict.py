import numpy as np
from tensorflow.python.keras.models import load_model


# Load the trained model
model = load_model("models/baseline_cnn_mitbih_final_model.h5")

# Define the mapping for decoding labels to categories
label_to_category = {
    0: 'Normal beats', 
    1: 'Supraventricular ectopic beats', 
    2: 'Ventricular ectopic beats', 
    3: 'Fusion beats', 
    4: 'Unknown beats'}

def predict(input_data):
    prediction = model.predict(input_data)
    print(prediction, prediction.shape)
    predicted_class = np.argmax(prediction, axis=-1)
    print(predicted_class)
    predicted_class = label_to_category[predicted_class[0]]
    return predicted_class

# Example input data (replace with actual ECG segment)
input_data = np.random.rand(1, 187)  # Simulating one ECG segment of 187 time steps
predicted_class = predict(input_data)

print(f"Predicted Class: {predicted_class}")
