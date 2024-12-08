import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
st.title("Water Potability Prediction")
st.write("This app predicts whether water is potable (safe for drinking) based on its properties.")

@st.cache_data
def load_data():
    water_data = pd.read_csv('water_potability.csv')
    return water_data

water_data = load_data()

# Preprocessing
X = water_data.drop(columns='Potability')
y = water_data.Potability

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0, test_size=0.2)

# Train a Random Forest Classifier
RF = RandomForestClassifier(random_state=42)
RF.fit(X_train, y_train)

# GridSearchCV setup
params_RF = {
    "min_samples_split": [2, 6],
    "min_samples_leaf": [1, 4],
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy"]
}
cv_method = StratifiedKFold(n_splits=3)
grid_search = GridSearchCV(
    estimator=RF,
    param_grid=params_RF,
    cv=cv_method,
    verbose=1,
    n_jobs=-1,
    scoring="accuracy",
    return_train_score=True
)
grid_search.fit(X_train, y_train)

# Use the best estimator
best_estimator = grid_search.best_estimator_

# Test the model
y_pred_best = best_estimator.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred_best) * 100, 2)

print(f"Model Accuracy: **{accuracy}%**")

# Display classification report
report = classification_report(y_test, y_pred_best, output_dict=True)
print("Classification Report:")
print(pd.DataFrame(report).transpose())

# User input for prediction
st.header("Enter Water Properties for Prediction")
with st.form("prediction_form"):
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=7.0)
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=500.0, value=150.0)
    solids = st.number_input("Solids (ppm)", min_value=0.0, max_value=50000.0, value=10000.0)
    chloramines = st.number_input("Chloramines (mg/L)", min_value=0.0, max_value=10.0, value=5.0)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, max_value=500.0, value=250.0)
    conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, max_value=1000.0, value=400.0)
    organic_carbon = st.number_input("Organic Carbon (mg/L)", min_value=0.0, max_value=30.0, value=10.0)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, max_value=150.0, value=80.0)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=3.0)
    submit = st.form_submit_button("Predict")

if submit:
    # Scale the user input
    user_input = scaler.transform([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Make prediction
    prediction = best_estimator.predict(user_input)[0]
    result = "Potable" if prediction == 1 else "Not Potable"
    
    # Display the result
    st.subheader("Prediction Result")
    st.write(f"The water is predicted to be: **{result}**")
