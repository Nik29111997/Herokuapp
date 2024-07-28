from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('fish_weight_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'Predicted fish weight: {output} grams')

if __name__ == "__main__":
    app.run(debug=True)