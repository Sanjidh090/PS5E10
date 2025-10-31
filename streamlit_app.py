import streamlit as st
import xgboost as xgb
import joblib
import numpy as np
import pandas as pd
import scipy.stats
import random

# Set page config
st.set_page_config(
    page_title="Road Safety Game",
    page_icon="🚗",
    layout="wide"
)

# ============================================================================
# Helper Classes and Functions (from the training notebook)
# ============================================================================

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
    """
    def __init__(self, f_func):
        self.f_func = f_func
        self.sigma = 0.05

    def __call__(self, X):
        """Makes instances of the class callable."""
        mu = self.f_func(X)
        sigma = self.sigma
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a) + sigma*(phi_a-phi_b) + 1 - Phi_b

# ============================================================================
# Load Model and Artifacts
# ============================================================================

@st.cache_resource
def load_model_and_artifacts():
    """Load the pre-trained model and preprocessing artifacts."""
    try:
        model = joblib.load('./xgboost_model.pkl')
        clipper = Clipper(f)  # Recreate clipper with our function
        prep_artifacts = joblib.load('./preprocessing_artifacts.pkl')
        return model, clipper, prep_artifacts
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, clipper, prep_artifacts = load_model_and_artifacts()

# ============================================================================
# Preprocessing Functions
# ============================================================================

def preprocess_single_road(road_data, prep_artifacts, clipper):
    """
    Preprocess a single road's data to match the training pipeline.
    
    Args:
        road_data: dict with keys like 'curvature', 'lighting', 'weather', etc.
        prep_artifacts: preprocessing artifacts from training
        clipper: Clipper object for feature engineering
    
    Returns:
        DataFrame ready for prediction
    """
    # Create a dataframe from the road data
    df = pd.DataFrame([road_data])
    
    # 1. Frequency encoding for ALL features in freq_maps (not just cat_cols)
    for col in prep_artifacts['freq_maps'].keys():
        if col in df.columns:
            freq = prep_artifacts['freq_maps'][col]
            mean_freq = prep_artifacts['freq_means'][col]
            df[f"{col}_freq"] = df[col].map(freq).fillna(mean_freq)
    
    # 2. Binning numeric features
    num_cols = prep_artifacts.get('num_cols', [])
    for col in num_cols:
        if col in df.columns:
            for q in [5, 10, 15]:
                bin_col = f"{col}_bin{q}"
                if bin_col in prep_artifacts['bin_edges']:
                    bins = prep_artifacts['bin_edges'][bin_col]
                    df[bin_col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
    
    # 3. Map specific columns
    map_col = "num_reported_accidents"
    if map_col in df.columns and 'map_num_reported' in prep_artifacts:
        map_num_reported = prep_artifacts['map_num_reported']
        df[map_col] = df[map_col].map(map_num_reported)
    
    # 4. Drop unnecessary columns
    cols_to_remove = prep_artifacts.get('cols_to_remove', [])
    df.drop(columns=[col for col in cols_to_remove if col in df.columns], inplace=True)
    
    # 5. Convert categorical columns (only the ones that remain after dropping)
    cat_cols = prep_artifacts.get('cat_cols', [])
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    # 6. Apply clipping
    df["curvature_clipped"] = df.apply(clipper, axis=1)
    
    return df

def predict_accident_risk(road_data, model, prep_artifacts, clipper):
    """
    Predict accident risk for a given road configuration.
    
    Args:
        road_data: dict with road features
        model: trained XGBoost model
        prep_artifacts: preprocessing artifacts
        clipper: Clipper object
    
    Returns:
        float: predicted accident risk
    """
    try:
        df_processed = preprocess_single_road(road_data, prep_artifacts, clipper)
        
        # Ensure columns are in the same order as expected by the model
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            df_processed = df_processed[expected_features]
        
        prediction = model.predict(df_processed)
        return float(prediction[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0

# ============================================================================
# Generate Random Road Scenarios
# ============================================================================

def generate_random_road():
    """Generate a random road configuration based on actual training features."""
    weather_options = ['clear', 'rainy', 'foggy', 'snowy']
    lighting_options = ['daylight', 'night']
    road_type_options = ['highway', 'rural', 'urban', 'residential']
    time_of_day_options = ['morning', 'afternoon', 'evening', 'night']
    
    road = {
        'curvature': round(random.uniform(0.1, 0.9), 2),
        'speed_limit': random.choice([30, 40, 50, 60, 70, 80]),
        'num_reported_accidents': random.randint(0, 7),
        'weather': random.choice(weather_options),
        'lighting': random.choice(lighting_options),
        'public_road': random.choice([0, 1]),
        'holiday': random.choice([0, 1]),
        'school_season': random.choice([0, 1]),
        'road_type': random.choice(road_type_options),
        'time_of_day': random.choice(time_of_day_options),
        'num_lanes': random.randint(1, 4)
    }
    return road

# ============================================================================
# Initialize Session State
# ============================================================================

if 'score' not in st.session_state:
    st.session_state.score = 0
if 'total_games' not in st.session_state:
    st.session_state.total_games = 0
if 'road_1' not in st.session_state:
    st.session_state.road_1 = generate_random_road()
if 'road_2' not in st.session_state:
    st.session_state.road_2 = generate_random_road()
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'last_choice' not in st.session_state:
    st.session_state.last_choice = None

# ============================================================================
# Main App Layout
# ============================================================================

st.title("🚗 Road Safety Game - Pick the Safer Road!")

st.markdown("""
Welcome to the **Road Safety Game**! Test your intuition about road safety by choosing 
the safer route between two roads. After each choice, see how your intuition compares 
to our AI model trained on real accident data.
""")

# Display score
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Score", st.session_state.score)
with col2:
    st.metric("Games Played", st.session_state.total_games)
with col3:
    if st.session_state.total_games > 0:
        accuracy = (st.session_state.score / st.session_state.total_games) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")

st.markdown("---")

# ============================================================================
# Game Section
# ============================================================================

st.header("🎮 Pick the Safer Road")

# Display two roads side by side
col_road1, col_road2 = st.columns(2)

with col_road1:
    st.subheader("🛣️ Road 1")
    road_1 = st.session_state.road_1
    st.write(f"**Curvature:** {road_1['curvature']}")
    st.write(f"**Speed Limit:** {road_1['speed_limit']} mph")
    st.write(f"**Number of Lanes:** {road_1['num_lanes']}")
    st.write(f"**Weather:** {road_1['weather'].title()}")
    st.write(f"**Lighting:** {road_1['lighting'].title()}")
    st.write(f"**Road Type:** {road_1['road_type'].title()}")
    st.write(f"**Time of Day:** {road_1['time_of_day'].title()}")
    st.write(f"**Public Road:** {'Yes' if road_1['public_road'] else 'No'}")
    st.write(f"**Holiday:** {'Yes' if road_1['holiday'] else 'No'}")
    st.write(f"**School Season:** {'Yes' if road_1['school_season'] else 'No'}")
    st.write(f"**Reported Accidents:** {road_1['num_reported_accidents']}")

with col_road2:
    st.subheader("🛣️ Road 2")
    road_2 = st.session_state.road_2
    st.write(f"**Curvature:** {road_2['curvature']}")
    st.write(f"**Speed Limit:** {road_2['speed_limit']} mph")
    st.write(f"**Number of Lanes:** {road_2['num_lanes']}")
    st.write(f"**Weather:** {road_2['weather'].title()}")
    st.write(f"**Lighting:** {road_2['lighting'].title()}")
    st.write(f"**Road Type:** {road_2['road_type'].title()}")
    st.write(f"**Time of Day:** {road_2['time_of_day'].title()}")
    st.write(f"**Public Road:** {'Yes' if road_2['public_road'] else 'No'}")
    st.write(f"**Holiday:** {'Yes' if road_2['holiday'] else 'No'}")
    st.write(f"**School Season:** {'Yes' if road_2['school_season'] else 'No'}")
    st.write(f"**Reported Accidents:** {road_2['num_reported_accidents']}")

# User makes a choice
st.markdown("### Which road do you think is safer?")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn1:
    if st.button("🛣️ Road 1 is Safer", use_container_width=True, type="primary"):
        st.session_state.last_choice = 1
        st.session_state.show_result = True
        st.rerun()

with col_btn2:
    if st.button("🛣️ Road 2 is Safer", use_container_width=True, type="primary"):
        st.session_state.last_choice = 2
        st.session_state.show_result = True
        st.rerun()

# Show results if a choice was made
if st.session_state.show_result and model is not None:
    st.markdown("---")
    st.header("📊 Results")
    
    # Get predictions
    risk_1 = predict_accident_risk(road_1, model, prep_artifacts, clipper)
    risk_2 = predict_accident_risk(road_2, model, prep_artifacts, clipper)
    
    # Display predictions
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Road 1 Risk Score", f"{risk_1:.4f}")
    with col_r2:
        st.metric("Road 2 Risk Score", f"{risk_2:.4f}")
    
    # Determine which road is actually safer
    safer_road = 1 if risk_1 < risk_2 else 2
    user_correct = st.session_state.last_choice == safer_road
    
    # Update score and total games
    if st.session_state.show_result:
        st.session_state.total_games += 1
        if user_correct:
            st.session_state.score += 1
    
    # Display result
    if user_correct:
        st.success(f"🎉 **Correct!** Road {safer_road} is indeed safer with a lower risk score!")
    else:
        st.error(f"❌ **Not quite.** Road {safer_road} is actually safer with a lower risk score.")
    
    st.info(f"💡 **Lower risk scores indicate safer roads.** Road {safer_road} has a risk score of {min(risk_1, risk_2):.4f} compared to {max(risk_1, risk_2):.4f}")

# Next round button
with col_btn3:
    if st.button("🔄 New Roads", use_container_width=True):
        st.session_state.road_1 = generate_random_road()
        st.session_state.road_2 = generate_random_road()
        st.session_state.show_result = False
        st.session_state.last_choice = None
        st.rerun()

st.markdown("---")

# ============================================================================
# Interactive Feature Explorer
# ============================================================================

st.header("🔍 Road Safety Explorer")
st.markdown("Adjust the road features below to see how they affect the predicted accident risk.")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    exp_curvature = st.slider("Curvature", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    exp_speed_limit = st.slider("Speed Limit (mph)", min_value=30, max_value=80, value=50, step=10)
    exp_num_lanes = st.slider("Number of Lanes", min_value=1, max_value=4, value=2)
    exp_num_accidents = st.slider("Number of Reported Accidents", min_value=0, max_value=7, value=2)
    exp_weather = st.selectbox("Weather Condition", ['clear', 'rainy', 'foggy', 'snowy'])
    exp_lighting = st.selectbox("Lighting Condition", ['daylight', 'night'])

with col_exp2:
    exp_road_type = st.selectbox("Road Type", ['highway', 'rural', 'urban', 'residential'])
    exp_time_of_day = st.selectbox("Time of Day", ['morning', 'afternoon', 'evening', 'night'])
    exp_public_road = st.checkbox("Public Road", value=True)
    exp_holiday = st.checkbox("Holiday", value=False)
    exp_school_season = st.checkbox("School Season", value=True)

# Create custom road from user inputs
custom_road = {
    'curvature': exp_curvature,
    'speed_limit': exp_speed_limit,
    'num_lanes': exp_num_lanes,
    'num_reported_accidents': exp_num_accidents,
    'weather': exp_weather,
    'lighting': exp_lighting,
    'road_type': exp_road_type,
    'time_of_day': exp_time_of_day,
    'public_road': 1 if exp_public_road else 0,
    'holiday': 1 if exp_holiday else 0,
    'school_season': 1 if exp_school_season else 0
}

if model is not None:
    custom_risk = predict_accident_risk(custom_road, model, prep_artifacts, clipper)
    
    st.markdown("### Predicted Risk Score")
    
    # Create a visual indicator
    risk_color = "green" if custom_risk < 0.1 else "orange" if custom_risk < 0.2 else "red"
    risk_label = "Low Risk" if custom_risk < 0.1 else "Medium Risk" if custom_risk < 0.2 else "High Risk"
    
    st.metric("Risk Score", f"{custom_risk:.4f}", risk_label)
    
    st.progress(min(custom_risk, 1.0))
    
    st.markdown(f"""
    **Risk Level:** :{risk_color}[{risk_label}]
    
    💡 **Tip:** Lower values indicate safer road conditions. Try changing different features to see 
    how they impact the overall safety prediction!
    """)

st.markdown("---")

# ============================================================================
# About Section
# ============================================================================

with st.expander("ℹ️ About This App"):
    st.markdown("""
    ### How It Works
    
    This interactive application uses a machine learning model trained on real road accident data 
    to predict the relative safety of different road conditions. The model considers multiple factors:
    
    - **Curvature**: How curved or winding the road is (higher = more curved)
    - **Speed Limit**: The maximum allowed speed on the road
    - **Number of Lanes**: Width and capacity of the road
    - **Weather**: Current weather conditions affecting visibility and traction
    - **Lighting**: Whether it's during daylight or at night
    - **Road Type**: Classification of the road (highway, rural, urban, residential)
    - **Time of Day**: Specific time period affecting traffic patterns
    - **Public Road**: Whether it's a public or private road
    - **Holiday**: Whether it's a holiday period with different traffic patterns
    - **School Season**: Whether schools are in session
    - **Historical Accidents**: Number of previously reported accidents on this road
    
    ### The Model
    
    The prediction model is an XGBoost Regressor trained on the Playground Series S5E10 dataset 
    from Kaggle. It uses advanced feature engineering and preprocessing techniques to make accurate 
    predictions about road safety.
    
    ### Challenge
    
    This app was created as part of the Stack Overflow and Kaggle Code Scientist challenge, 
    which involves both building a predictive model and creating an interactive web application 
    to explore the data.
    
    ### Tips for Playing
    
    - Think about how different factors might combine to affect safety
    - Consider that some factors (like speed limit and curvature) might interact
    - Night driving combined with bad weather is typically more dangerous
    - Use the explorer section to understand how individual features affect risk
    """)

st.markdown("---")
st.caption("Built with Streamlit | Model trained on Kaggle Playground Series S5E10 data")
