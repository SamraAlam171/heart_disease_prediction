
import pickle
import pandas as pd
import streamlit as st

# Load the trained pipeline
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction System")
st.write("Enter the information below to get a model prediction.")

age = st.number_input(
    "Age (years)", min_value=1, max_value=120, value=45
)

sex = st.selectbox("Sex", ["Female", "Male"])

cp = st.selectbox(
    "Chest Pain Type",
    [
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic"
    ]
)

thalch = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=1, max_value=250, value=150
)

if st.button("Predict"):
    user_input = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "thalch": thalch
    }])

    prediction = model.predict(user_input)[0]

    if prediction == 0:
        st.success("The model predicts: No heart disease recorded.")
    else:
        st.warning("The model predicts: Heart disease recorded.")

    st.caption(
        "Educational demonstration only. This is not a medical diagnosis."
    )
