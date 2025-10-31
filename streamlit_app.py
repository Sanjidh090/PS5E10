import streamlit as st
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained model and preprocessing artifacts
model = pickle.load(open('xgboost_model.pkl', 'rb'))
scaler = pickle.load(open('preprocessing_artifacts.pkl', 'rb'))
feature_clipper = pickle.load(open('feature_engineering_clipper.pkl', 'rb'))

# Function to predict accident risk
def predict_accident(features):
    # Apply necessary preprocessing steps to the input data
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    return prediction[0]

# Streamlit Layout
st.title("Pick the Safer Road Game")

st.write("""
    In this game, you will be presented with two roads and their features. 
    Use your intuition to select the safer road. After you make your choice, 
    we will show you the model's prediction.
""")

# Present two road options with different features
road_1 = {
    'lanes': 3,
    'curvature': 0.35,
    'time_of_day': 'night',
    'accident_risk': 0.25
}

road_2 = {
    'lanes': 2,
    'curvature': 0.75,
    'time_of_day': 'daylight',
    'accident_risk': 0.55
}

# Display road features
st.subheader("Road 1")
st.write(f"Lanes: {road_1['lanes']}")
st.write(f"Curvature: {road_1['curvature']}")
st.write(f"Time of day: {road_1['time_of_day']}")

st.subheader("Road 2")
st.write(f"Lanes: {road_2['lanes']}")
st.write(f"Curvature: {road_2['curvature']}")
st.write(f"Time of day: {road_2['time_of_day']}")

# User selects the safer road
user_choice = st.radio(
    "Which road do you think is safer?",
    ("Road 1", "Road 2")
)

# Predefined features for prediction
road_1_features = [road_1['lanes'], road_1['curvature'], 1 if road_1['time_of_day'] == 'night' else 0]
road_2_features = [road_2['lanes'], road_2['curvature'], 1 if road_2['time_of_day'] == 'night' else 0]

# Predict accident risks for both roads
road_1_risk = predict_accident(road_1_features)
road_2_risk = predict_accident(road_2_features)

# Feedback based on user choice
if user_choice == "Road 1":
    user_prediction = road_1_risk
else:
    user_prediction = road_2_risk

# Display the model's prediction after user makes a choice
st.write(f"\n**Model's prediction:**")

if road_1_risk < road_2_risk:
    st.write("**Road 1** is safer according to the model!")
else:
    st.write("**Road 2** is safer according to the model!")

# Check if the user made the correct choice
if user_choice == "Road 1" and road_1_risk < road_2_risk:
    st.success("You were correct! Road 1 is safer.")
elif user_choice == "Road 2" and road_2_risk < road_1_risk:
    st.success("You were correct! Road 2 is safer.")
else:
    st.error("Oops! You were wrong. Try again!")

# Optionally, allow the user to modify road features and see how it affects predictions
st.subheader("Adjust road features to see how they affect safety predictions:")

lanes = st.slider("Number of lanes", min_value=1, max_value=5, value=3)
curvature = st.slider("Curvature", min_value=0.0, max_value=1.0, value=0.5)
time_of_day = st.selectbox("Time of day", ["Daylight", "Night"])

# Convert the selected features into a prediction
adjusted_features = [lanes, curvature, 1 if time_of_day == 'Night' else 0]
adjusted_risk = predict_accident(adjusted_features)

st.write(f"Adjusted Road Risk: {adjusted_risk}")
