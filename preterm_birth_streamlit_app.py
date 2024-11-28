import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Load Model and Updated Scaler
@st.cache_resource  # Cache the model and scaler for performance
def load_resources():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'preterm_birth_rf_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_updated.pkl')

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)  # Use the updated scaler with 16 features
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()

model, scaler = load_resources()

# Define the 16 Feature Names (excluding 'Prenatal_Care_Visits')
features = ['Pregnancies', 'Age', 'BMI', 'BloodPressure', 'Glucose',
            'Smoking', 'Previous_Preterm_Births', 'Gestational_Diabetes',
            'Genetic_Factors', 'Environmental_Factors',
            'PCOS', 'HIV', 'Zika_infection', 'Thyroid', 'Autoimmune_disease', 'Kidney_disease']

# Prediction Function without Clipping
def make_prediction(input_data):
    try:
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        return prediction[0]  # Return raw prediction without clipping
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

# Function to Display Feature Importance
def plot_feature_importance():
    try:
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.gca().invert_yaxis()
        plt.xlabel("Importance Score")
        plt.title("Feature Importance")
        st.pyplot(plt)
        plt.close()  # Close the plot after rendering to avoid duplication in Streamlit
    except AttributeError:
        st.warning("Feature importance is not available for the loaded model.")

# Main Application
def main():
    st.title("Preterm Birth Prediction Model")

    # User input options
    st.write("### Choose Input Method")
    option = st.selectbox("Input data via", ["Manual Input", "Upload CSV File"])

    # Manual Input Option
    if option == "Manual Input":
        st.write("### Enter the required parameters manually:")
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        age = st.number_input("Age", min_value=10, max_value=50, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
        glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
        smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
        previous_preterm_births = st.number_input("Previous Preterm Births", min_value=0, max_value=10, value=0)
        gestational_diabetes = st.selectbox("Gestational Diabetes (0 = No, 1 = Yes)", [0, 1])
        genetic_factors = st.selectbox("Genetic Factors (0 = No, 1 = Yes)", [0, 1])
        environmental_factors = st.selectbox("Environmental Factors (0 = No, 1 = Yes)", [0, 1])
        pcos = st.number_input("PCOS Score", min_value=0.0, max_value=10.0, value=5.0)
        hiv = st.number_input("HIV Level", min_value=0.0, max_value=100.0, value=10.0)
        zika_infection = st.number_input("Zika Infection Level", min_value=0.0, max_value=50.0, value=5.0)
        thyroid = st.number_input("Thyroid Level", min_value=0.0, max_value=5.0, value=2.5)
        autoimmune_disease = st.number_input("Autoimmune Disease Score", min_value=0.0, max_value=10.0, value=5.0)
        kidney_disease = st.number_input("Kidney Disease Level", min_value=0.0, max_value=100.0, value=50.0)

        if st.button("Predict"):
            # Make sure input data matches the features defined
            input_data = np.array([[pregnancies, age, bmi, blood_pressure, glucose, smoking,
                                    previous_preterm_births, gestational_diabetes,
                                    genetic_factors, environmental_factors, pcos, hiv, zika_infection,
                                    thyroid, autoimmune_disease, kidney_disease]])

            prediction = make_prediction(input_data)
            st.success(f"Predicted Gestational Period: {prediction:.2f} days")

            # Display Feature Importance
            st.write("### Feature Importance")
            plot_feature_importance()

    # CSV Upload Option
    elif option == "Upload CSV File":
        st.write("### Upload your CSV file for batch predictions below:")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded CSV file:")
                st.write(data.head())

                # Check if all required columns are in the uploaded file
                required_columns = set(features)
                if required_columns.issubset(data.columns):
                    # Scale and predict
                    input_data_scaled = scaler.transform(data[features])
                    predictions = model.predict(input_data_scaled)

                    # Add predictions to DataFrame
                    data['Predicted_Gestational_Period_Days'] = predictions

                    st.write("Predictions:")
                    st.write(data[features + ['Predicted_Gestational_Period_Days']])

                    # Display Feature Importance
                    st.write("### Feature Importance")
                    plot_feature_importance()

                    # Download predictions as CSV
                    csv_data = data.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions as CSV", data=csv_data, file_name='predictions.csv')
                else:
                    missing_cols = required_columns - set(data.columns)
                    st.error(f"The uploaded file is missing required columns: {', '.join(missing_cols)}")
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")

if __name__ == "__main__":
    main()
