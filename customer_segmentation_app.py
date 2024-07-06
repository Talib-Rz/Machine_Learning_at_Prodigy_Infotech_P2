from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
with open('DecisionTreeClassifier_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        annual_income = int(request.form['annual_income'])
        spending_score = int(request.form['spending_score'])

        # Create a DataFrame with the feature names
        feature_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        input_data = pd.DataFrame([[age, annual_income, spending_score]], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "Good Customer for Life_Time"
        elif prediction[0] == 2:
            result = "Customers, They need more attention"
        elif prediction[0] == 0:
            result = "Customers are mediocre, sometimes they will buy.. sometimes they won't"
        else:
            result = "Unexpected prediction value."

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text="An error occurred: {}".format(e))

if __name__ == "__main__":
    app.run(debug=True)
