from wsgiref import simple_server
from flask import Flask, request, render_template
from flask_cors import CORS
import pickle
import bz2
import numpy as np

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

# Correcting the file paths
scalarobject = bz2.BZ2File("Model/standardScalar.pkl", "rb")  # Use forward slashes
scaler = pickle.load(scalarobject)
modelforpred = bz2.BZ2File("Model/modelForPrediction.pkl", "rb")
model = pickle.load(modelforpred)

# Route for homepage
@app.route('/')
def root():
    return render_template('home.html')

# Route for Single data point prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':
        # Extract data from the form
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        # Transforming input data using the scaler
        new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        # Predicting using the model
        predict = model.predict(new_data)

        # Determining result based on the prediction
        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'

        # Rendering prediction result in the HTML
        return render_template('single_prediction.html', result=result)

    else:
        return render_template('home.html')

# Running the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
