import streamlit as st
import pandas as pd
import joblib

# Load the trained model
with open('DecisionTreeClassifier_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

st.title("Customer Classification Prediction")

# Function to make prediction
def predict_customer_class(age, annual_income, spending_score):
    feature_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    input_data = pd.DataFrame([[age, annual_income, spending_score]], columns=feature_names)
    
    # Make prediction and add debug information
    prediction = model.predict(input_data)
    st.write(f"Model prediction raw output: {prediction}")

    if prediction[0] == 1:
        return "Good Customer for Life_Time"
    elif prediction[0] == 2:
        return "Customers, They need more attention"
    else:
        return f"Unexpected prediction value: {prediction[0]}, Customers are mediocre, sometimes they will buy.. sometimes they won't"

# Streamlit input widgets
age = st.number_input("Age", min_value=0, max_value=120, value=25)
annual_income = st.number_input("Annual Income (k$)", min_value=0, max_value=1000, value=100)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=80)

# Prediction button
if st.button("Predict"):
    try:
        result = predict_customer_class(age, annual_income, spending_score)
        st.success(result)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# To run this app, save the code in a file named `app.py` and run the following command:
# streamlit run app.py
