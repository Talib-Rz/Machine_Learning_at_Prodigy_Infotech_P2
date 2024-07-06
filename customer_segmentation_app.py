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
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        return "Good Customer for Life_Time"
    elif prediction[0] == 2:
        return "Customers, They need more attention"
    elif prediction[0] == 0:
        return "Customers are mediocre, sometimes they will buy.. sometimes they won't"
    else:
        return "Unexpected prediction value."

# Streamlit input widgets
age = st.number_input("Age", min_value=0, max_value=120, value=30)
annual_income = st.number_input("Annual Income (k$)", min_value=0, max_value=1000, value=50)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Prediction button
if st.button("Predict"):
    try:
        result = predict_customer_class(age, annual_income, spending_score)
        st.success(result)
    except Exception as e:
        st.error(f"An error occurred: {e}")
