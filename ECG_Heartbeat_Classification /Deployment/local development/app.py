import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify


model = load_model("baseline_cnn_mitbih_final_model.h5")
# model.summary()

label_to_category = {
    0: 'Normal beats', 
    1: 'Supraventricular ectopic beats', 
    2: 'Ventricular ectopic beats', 
    3: 'Fusion beats', 
    4: 'Unknown beats'}


app = Flask(__name__)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        sample = request.get_json()
        # Reshape for the model (187 time steps, 1 channel)
        sample_arr = np.array(sample["sample"]).reshape(1, 187, 1) 
      
        prediction = model.predict(sample_arr)
        predicted_class = np.argmax(prediction, axis=-1)[0] 
        
        # Decode the predicted class to its corresponding label
        decoded_label = label_to_category.get(predicted_class, "Unknown")  
        result = {"prediction": decoded_label}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)