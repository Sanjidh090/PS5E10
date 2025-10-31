import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats  # We don't call it, but the clipper.pkl file needs it
import joblib

# --- 1. LOAD MODELS AND ARTIFACTS ---
# These 3 files MUST be in your GitHub repo
try:
    model = joblib.load("xgboost_model.pkl")
    clipper = joblib.load("feature_engineering_clipper.pkl")
    artifacts = joblib.load("preprocessing_artifacts.pkl")
    
    # Extract artifacts for easier use
    freq_maps = artifacts['freq_maps']
    freq_means = artifacts['freq_means']
    bin_edges = artifacts['bin_edges']
    map_num_reported = artifacts['map_num_reported']
    cat_cols = artifacts['cat_cols']
    num_cols = artifacts['num_cols']
    cols_to_remove = artifacts['cols_to_remove']
    
except FileNotFoundError:
    st.error("Error: Model/artifact .pkl files not found. Please upload all 3 files to your GitHub repo.")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


# --- 2. DEFINE THE INFERENCE AND PREPROCESSING FUNCTIONS ---

# This 'f' function MUST be identical to the one in your notebook
# The 'clipper' object needs it to exist in this scope
def f(X):
    return \
    0.35 * X["curvature"] + \
    0.05 * int(X["lighting"] == "night") + \
    0.1 * int(X["weather"] != "clear") + \
    0.35 * int(X["speed_limit"] >= 60) + \
    0.2 * int(X["num_reported_accidents"] > 2)

# This new function replicates your preprocessing pipeline for a single input
def preprocess_for_inference(input_df, clipper, artifacts):
    """
    Preprocesses a single-row DataFrame from user input
    using the loaded artifacts.
    """
    df = input_df.copy()

    # 1. Frequency Encoding
    for col in artifacts['cat_cols']:
        if col in df.columns:
            freq_map = artifacts['freq_maps'][col]
            mean_val = artifacts['freq_means'][col]
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(mean_val)

    # 2. Binning Numeric Features
    for col in artifacts['num_cols']:
        for q in [5, 10, 15]:
            bin_col_name = f"{col}_bin{q}"
            if bin_col_name in artifacts['bin_edges']:
                bins = artifacts['bin_edges'][bin_col_name]
                # Ensure bin edges are finite for pd.cut
                bins[0] = -np.inf
                bins[-1] = np.inf
                df[bin_col_name] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True, right=True)
                df[bin_col_name] = df[bin_col_name].fillna(0) # Fill NaNs

    # 3. Mapping 'num_reported_accidents'
    map_col = "num_reported_accidents"
    if map_col in df.columns and artifacts['map_num_reported']:
        df[map_col] = df[map_col].map(artifacts['map_num_reported']).fillna(0) # FillNa if unknown input

    # 4. Drop unnecessary columns
    cols_to_drop = [col for col in artifacts['cols_to_remove'] if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # 5. Apply Scipy-based clipping
    df["curvature_clipped"] = df.apply(clipper, axis=1)

    # 6. Ensure categorical columns are 'category' type
    for col in artifacts['cat_cols']:
        if col in df.columns:
            # Use categories from the artifact map to ensure consistency
            if col in artifacts['freq_maps']:
                known_categories = list(artifacts['freq_maps'][col].index)
                df[col] = pd.Categorical(df[col], categories=known_categories)
            else:
                 df[col] = df[col].astype("category")
            
    return df


# --- 3. BUILD THE STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Road Accident Risk Predictor", layout="wide")
st.title("🚧 Road Accident Risk Predictor")

st.sidebar.header("Enter Road & Weather Conditions:")

# --- Create input fields ---
# I'm adding all features I see in your 'f' function and artifact list
# You MUST add any other columns your model needs.

st.sidebar.subheader("Key Features")
curvature = st.sidebar.number_input("Road Curvature (float, e.g., 0.05)", value=0.05, step=0.01)
lighting = st.sidebar.selectbox("Lighting Conditions", ["day", "night", "dusk", "dawn"]) # From 'f'
weather = st.sidebar.selectbox("Weather Conditions", ["clear", "rainy", "foggy", "snowy", "other"]) # From 'f'
speed_limit = st.sidebar.slider("Speed Limit (km/h)", 20, 120, 60) # From 'f'
num_reported_accidents = st.sidebar.slider("Historical Accidents (in this area)", 0, 10, 2) # From 'f'

st.sidebar.subheader("Other Features")
# Add all other columns from your original 'train.csv' that are NOT in the 'remove' list
# I am guessing based on the 'remove' list
id = "dummy_id_999" # 'id' is in the remove list, so just use a placeholder
time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"]) # In remove list? If not, make it an input
num_lanes = st.sidebar.slider("Number of Lanes", 1, 6, 2) # In remove list?
road_type = st.sidebar.selectbox("Road Type", ["Highway", "Street", "Rural"]) # In remove list?
road_signs_present = st.sidebar.selectbox("Road Signs Present?", ["Yes", "No"]) # In remove list?

# --- 4. PREDICT AND DISPLAY RESULTS ---

if st.sidebar.button("Predict Risk Score"):
    
    # 1. Create a dictionary of all inputs
    input_data = {
        # Features from 'f' function
        'curvature': curvature,
        'lighting': lighting,
        'weather': weather,
        'speed_limit': speed_limit,
        'num_reported_accidents': num_reported_accidents,
        
        # Other features (placeholders or real)
        'id': id, 
        'time_of_day': time_of_day,
        'num_lanes': num_lanes,
        'road_type': road_type,
        'road_signs_present': road_signs_present,
        
        # Add ANY other columns your model expects
        # e.g., 'driver_age': 30, (if you had this)
    }

    # 2. Convert to a single-row DataFrame
    input_df = pd.DataFrame([input_data])

    try:
        # 3. Preprocess the data
        processed_df = preprocess_for_inference(input_df.copy(), clipper, artifacts)
        
        # 4. Make prediction
        # Get feature names from the loaded model
        model_features = model.get_booster().feature_names
        
        # Ensure processed_df has all columns in the right order, filling missing ones with 0 or a default
        processed_df_final = pd.DataFrame(columns=model_features)
        processed_df_final = pd.concat([processed_df_final, processed_df])
        
        # Handle type conversion for categorical columns that might be all NaN
        for col in processed_df_final.select_dtypes(include=['category']).columns:
            if processed_df_final[col].isnull().all():
                 # If a category is all NaN, set it to a known category or drop?
                 # Easiest is to convert to object and let XGB handle it
                 processed_df_final[col] = processed_df_final[col].astype(object).fillna("Unknown")
                 processed_df_final[col] = processed_df_final[col].astype("category")

        # Fill any remaining NaNs (e.g., from new freq/bin columns)
        processed_df_final = processed_df_final.fillna(0)
        
        # Filter to only model features
        processed_df_final = processed_df_final[model_features]

        prediction = model.predict(processed_df_final)
        risk_score = prediction[0]

        # 5. Display the result
        st.subheader(f"Predicted Risk Score: {risk_score:.4f}")
        st.progress(float(risk_score)) # Assumes score is between 0 and 1
        
        if risk_score > 0.75:
            st.error("High Risk: Extreme caution advised.")
        elif risk_score > 0.5:
            st.warning("Moderate Risk: Be vigilant.")
        else:
            st.success("Low Risk: Conditions appear safe.")

    except Exception as e:
        st.error("An error occurred during preprocessing or prediction:")
        st.exception(e)
