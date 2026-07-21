import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("decision_tree_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🍷 Wine Quality Prediction")
st.markdown("""
Predict whether a **Red Wine** is **GOOD** or **BAD**
using a Machine Learning model trained on physicochemical properties.
""")

st.divider()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Wine Details")

fixed_acidity = st.sidebar.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", value=0.70)
citric_acid = st.sidebar.number_input("Citric Acid", value=0.00)
residual_sugar = st.sidebar.number_input("Residual Sugar", value=1.90)
chlorides = st.sidebar.number_input("Chlorides", value=0.076, format="%.3f")
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", value=34.0)
density = st.sidebar.number_input("Density", value=0.9978, format="%.4f")
ph = st.sidebar.number_input("pH", value=3.51)
sulphates = st.sidebar.number_input("Sulphates", value=0.56)
alcohol = st.sidebar.number_input("Alcohol", value=9.40)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Wine Quality"):

    sample = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol
    ]])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    confidence = np.max(probability) * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🍷 GOOD Quality Wine")
    else:
        st.error("🍷 BAD Quality Wine")

    st.metric("Confidence", f"{confidence:.2f}%")

    st.write("### Prediction Probabilities")

    st.progress(float(probability[1]))
    st.write(f"GOOD : **{probability[1]*100:.2f}%**")

    st.progress(float(probability[0]))
    st.write(f"BAD : **{probability[0]*100:.2f}%**")

st.divider()

# -----------------------------
# About
# -----------------------------
st.markdown("""
### 📌 About this Project

This project predicts wine quality using a **Decision Tree Classifier**
optimized with **GridSearchCV**.

### Model Performance

- Accuracy: **92.50%**
- Precision: **73.17%**
- Recall: **69.77%**
- F1-Score: **71.43%**

Developed using:

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
""")
