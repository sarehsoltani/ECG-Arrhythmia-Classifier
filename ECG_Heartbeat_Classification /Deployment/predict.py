import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

model = load_model("models/baseline_cnn_mitbih_final_model.h5")

# Define the mapping for decoding labels to categories
label_to_category = {
    1: 'Normal beats', 
    2: 'Supraventricular ectopic beats', 
    3: 'Ventricular ectopic beats', 
    4: 'Fusion beats', 
    5: 'Unknown beats'}


app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
    sample = request.get_json()

    # Ensure input is in the correct format for the model
    sample_array = np.array([list(sample.values())])

    y_pred = model.predict(sample_array)
    decoded_label = label_to_category.get(y_pred[0], "Unknown")

    result = {"type": decoded_label}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)