import numpy as np
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model("models/baseline_cnn_mitbih_final_model.h5")

# Define the mapping for decoding labels to categories
label_to_category = {
    1: 'Normal beats', 
    2: 'Supraventricular ectopic beats', 
    3: 'Ventricular ectopic beats', 
    4: 'Fusion beats', 
    5: 'Unknown beats'}

def predict(input_data):
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=-1)
    predicted_class = label_to_category[predicted_class[0]]
    return predicted_class

# Example input data (replace with actual ECG segment)
input_data = np.random.rand(1, 187)  # Simulating one ECG segment of 187 time steps
predicted_class = predict(input_data)

print(f"Predicted Class: {predicted_class}")
