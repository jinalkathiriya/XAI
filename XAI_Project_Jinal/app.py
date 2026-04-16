import streamlit as st
import pandas as pd
import shap
import joblib
import numpy as np
from lime import lime_tabular
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(__file__)

# title
st.title("Explainable AI System")
st.subheader("Diabetes Prediction with Explanation")

# load dataset
df = pd.read_csv(os.path.join(BASE_DIR,"dataset.csv"))
# split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# load model
model = joblib.load(os.path.join(BASE_DIR,"model.pkl"))

# input box
st.write("Enter patient details:")

preg = st.number_input("Pregnancies", 0, 20)
glu = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
ins = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# prediction button
if st.button("Predict"):

    input_data = np.array([[preg,glu,bp,skin,ins,bmi,dpf,age]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)[0][1]

    # prediction result
    if prediction[0] == 1:
        st.error("Diabetes Risk = YES")
    else:
        st.success("Diabetes Risk = NO")

    st.write("Risk Probability =", round(probability*100,2), "%")

    # risk level
    if probability > 0.7:
        risk_level = "HIGH"
        st.warning("Risk Level = HIGH")
    elif probability > 0.4:
        risk_level = "MODERATE"
        st.info("Risk Level = MODERATE")
    else:
        risk_level = "LOW"
        st.success("Risk Level = LOW")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_scaled)

    # FIX ERROR
    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
    else:
        contributions = shap_values[0]

    st.subheader("Why this prediction happened:")

    for feature, value in zip(X.columns, contributions):

        # ensure value is single number
        value = np.array(value).flatten()[0]

        print(type(value))
        print(value)
        print(np.shape(value))

        percent = abs(value) * 100

        if percent > 20:
            impact = "High impact"
        elif percent > 10:
            impact = "Moderate impact"
        else:
            impact = "Low impact"

        print(type(percent))
        print(percent)

        st.write(f"{feature} → {impact} ({percent:.2f}%)")

    # LIME explanation
    lime_exp = lime_tabular.LimeTabularExplainer(
        X_scaled,
        feature_names=X.columns,
        class_names=["No Diabetes","Diabetes"],
        mode="classification"
    )

    exp = lime_exp.explain_instance(
        input_scaled[0],
        model.predict_proba
    )

    st.subheader("Detailed Explanation (LIME):")

    for item in exp.as_list():
        st.write(item)

    # Health Suggestions
    st.subheader("Health Suggestions")

    if risk_level == "HIGH":

        st.error("High Risk Care Plan")

        st.write("Diet:")
        st.write("- Eat whole grains (brown rice, oats)")
        st.write("- Green vegetables (spinach, broccoli)")
        st.write("- High fiber foods")
        st.write("- Drink more water")

        st.write("Foods to Avoid:")
        st.write("- Sugary drinks")
        st.write("- White bread")
        st.write("- Fried foods")
        st.write("- Fast food")
        st.write("- Sweets")

        st.write("Exercise:")
        st.write("- Walking 30-45 minutes daily")
        st.write("- Yoga")
        st.write("- Light jogging")

        st.write("Lifestyle:")
        st.write("- Maintain healthy weight")
        st.write("- Sleep 7-8 hours")
        st.write("- Reduce stress")

    elif risk_level == "MODERATE":

        st.warning("Moderate Risk Care Plan")

        st.write("Diet:")
        st.write("- Balanced diet")
        st.write("- Fruits and vegetables")
        st.write("- Reduce sugar intake")

        st.write("Foods to Avoid:")
        st.write("- Limit sweets")
        st.write("- Avoid soft drinks")
        st.write("- Reduce oily food")

        st.write("Exercise:")
        st.write("- Walk 20-30 minutes daily")
        st.write("- Light exercise")

        st.write("Lifestyle:")
        st.write("- Regular health checkup")
        st.write("- Stay active")

    else:

        st.success("Low Risk Care Plan")

        st.write("Diet:")
        st.write("- Healthy balanced diet")
        st.write("- Fresh fruits")
        st.write("- Vegetables")

        st.write("Exercise:")
        st.write("- Daily walking")
        st.write("- Stay active")

        st.write("Lifestyle:")
        st.write("- Maintain healthy habits")
