import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats  # This is needed
import joblib

# --- 1. ADD CLASS AND FUNCTION DEFINITIONS HERE ---
# This is the fix. joblib.load() needs to see these definitions.

def f(X):
    """Function for feature engineering (for clipping)."""
    return \
    0.35 * X["curvature"] + \
    0.05 * int(X["lighting"] == "night") + \
    0.1 * int(X["weather"] != "clear") + \
    0.35 * int(X["speed_limit"] >= 60) + \
    0.2 * int(X["num_reported_accidents"] > 2)

class Clipper:
    """
    Class to handle the clipping logic.
    This object CAN be pickled by joblib.
    """
    def __init__(self, f_func):
        self.f_func = f_func
        self.sigma = 0.05

    def __call__(self, X):
        """Makes instances of the class callable."""
        mu = self.f_func(X)  # Apply the stored function 'f'
        sigma = self.sigma
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a) + sigma*(phi_a-phi_b) + 1 - Phi_b

# --- 2. LOAD MODELS AND ARTIFACTS ---
# Now this part will work
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


# --- 3. DEFINE THE INFERENCE AND PREPROCESSING FUNCTIONS ---

def preprocess_for_inference(input_df, clipper_obj, artifacts):
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
                bins[0] = -np.inf
                bins[-1] = np.inf
                df[bin_col_name] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True, right=True)
                df[bin_col_name] = df[bin_col_name].fillna(0)

    # 3. Mapping 'num_reported_accidents'
    map_col = "num_reported_accidents"
    if map_col in df.columns and artifacts['map_num_reported']:
        df[map_col] = df[map_col].map(artifacts['map_num_reported']).fillna(0)

    # 4. Drop unnecessary columns
    cols_to_drop = [col for col in artifacts['cols_to_remove'] if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # 5. Apply Scipy-based clipping
    # We pass the loaded 'clipper_obj' here
    df["curvature_clipped"] = df.apply(clipper_obj, axis=1)

    # 6. Ensure categorical columns are 'category' type
    for col in artifacts['cat_cols']:
        if col in df.columns:
            if col in artifacts['freq_maps']:
                known_categories = list(artifacts['freq_maps'][col].index)
                df[col] = pd.Categorical(df[col], categories=known_categories)
            else:
                 df[col] = df[col].astype("category")
            
    return df


# --- 4. BUILD THE STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Road Accident Risk Predictor", layout="wide")
st.title("🚧 Road Accident Risk Predictor")

st.sidebar.header("Enter Road & Weather Conditions:")

# --- Create input fields ---
st.sidebar.subheader("Key Features")
curvature = st.sidebar.number_input("Road Curvature (float, e.g., 0.05)", value=0.05, step=0.01)
lighting = st.sidebar.selectbox("Lighting Conditions", ["day", "night", "dusk", "dawn"])
weather = st.sidebar.selectbox("Weather Conditions", ["clear", "rainy", "foggy", "snowy", "other"])
speed_limit = st.sidebar.slider("Speed Limit (km/h)", 20, 120, 60)
num_reported_accidents = st.sidebar.slider("Historical Accidents (in this area)", 0, 10, 2)

st.sidebar.subheader("Other Features")
# These are the columns you removed, so we need inputs for them
# because your freq_maps/etc might still use them before they are dropped.
time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"]) 
num_lanes = st.sidebar.slider("Number of Lanes", 1, 6, 2)
road_type = st.sidebar.selectbox("Road Type", ["Highway", "Street", "Rural", "Intersection", "Other"]) 
road_signs_present = st.sidebar.selectbox("Road Signs Present?", ["Yes", "No"]) 
id = "dummy_id_999" # 'id' is in the remove list, so just use a placeholder

# --- 5. PREDICT AND DISPLAY RESULTS ---

if st.sidebar.button("Predict Risk Score"):
    
    # 1. Create a dictionary of all inputs
    input_data = {
        'curvature': curvature,
        'lighting': lighting,
        'weather': weather,
        'speed_limit': speed_limit,
        'num_reported_accidents': num_reported_accidents,
        
        'id': id, 
        'time_of_day': time_of_day,
        'num_lanes': num_lanes,
        'road_type': road_type,
        'road_signs_present': road_signs_present,
    }

    # 2. Convert to a single-row DataFrame
    input_df = pd.DataFrame([input_data])

    try:
        # 3. Preprocess the data
        # We pass the loaded 'clipper' object from step 2
        processed_df = preprocess_for_inference(input_df.copy(), clipper, artifacts)
        
        # 4. Make prediction
        model_features = model.get_booster().feature_names
        
        processed_df_final = pd.DataFrame(columns=model_features)
        processed_df_final = pd.concat([processed_df_final, processed_df])
        
        # --- START OF FIX ---
        
        # Handle NaNs in CATEGORICAL columns first
        for col in processed_df_final.select_dtypes(include=['category']).columns:
            # Add "Unknown" as a valid category if it doesn't exist
            if "Unknown" not in processed_df_final[col].cat.categories:
                processed_df_final[col] = processed_df_final[col].cat.add_categories("Unknown")
            
            # Fill any NaNs with our safe "Unknown" category
            processed_df_final[col] = processed_df_final[col].fillna("Unknown")

        # Now, fill all remaining NaNs (which are in NUMERIC columns) with 0
        processed_df_final = processed_df_final.fillna(0)
        
        # --- END OF FIX ---
        
        # Filter to only model features (ensures correct column order)
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
