# Road Safety Game - PS5E10 🚗

An interactive web application for the Stack Overflow and Kaggle Code Scientist Challenge. This project combines machine learning with an engaging game to explore road safety data.

## 🎮 Try the Game

[Live Demo on Streamlit Cloud](https://your-app-url-here.streamlit.app) _(Deploy to get URL)_

## 📋 About

This project was created for the **Playground Series S5E10** Kaggle competition and the Stack Overflow challenge. It features:

1. **Machine Learning Model**: XGBoost Regressor trained on road accident data
2. **Interactive Game**: "Pick the Safer Road" - test your intuition against AI predictions
3. **Feature Explorer**: Adjust road conditions and see how they affect safety predictions

## 🎯 Features

### Pick the Safer Road Game
- Compare two randomly generated roads with different characteristics
- Make predictions based on your intuition
- Get immediate feedback from the AI model
- Track your score and accuracy over multiple rounds

### Road Safety Explorer
- Adjust multiple road features using interactive controls:
  - Curvature
  - Speed limit
  - Number of lanes
  - Weather conditions
  - Lighting conditions
  - Road type
  - Time of day
  - Public/private road
  - Holiday status
  - School season
  - Historical accident count
- Real-time risk predictions
- Visual risk indicators (Low/Medium/High)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sanjidh090/PS5E10.git
   cd PS5E10
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

## 📦 Dependencies

- pandas
- numpy
- scipy
- xgboost
- joblib
- streamlit

## 🔧 Model Details

The prediction model uses:
- **Algorithm**: XGBoost Regressor
- **Training Data**: Kaggle Playground Series S5E10 dataset
- **Feature Engineering**: 
  - Frequency encoding for categorical variables
  - Binning for numeric features
  - Custom clipping function for curvature
  - Multiple interaction features

### Key Features Considered
- Road curvature
- Speed limits
- Weather conditions (clear, rainy, foggy, snowy)
- Lighting conditions (daylight, dim, night)
- Road type (highway, rural, urban, residential)
- Number of lanes
- Historical accident data
- Temporal factors (time of day, holidays, school season)
- Road classification (public/private)

## 📊 How It Works

1. **Data Preprocessing**: Raw road data goes through multiple preprocessing steps including frequency encoding, binning, and feature engineering
2. **Model Prediction**: The XGBoost model predicts accident risk scores (lower = safer)
3. **Interactive Gameplay**: Users compare two roads and test their intuition
4. **Real-time Feedback**: Immediate model predictions help users learn about road safety factors

## 🎓 What I Learned

Key challenges and insights from building this project:

1. **Model Deployment**: Converting a Kaggle notebook model to a production-ready Streamlit app
2. **Feature Engineering**: Recreating complex preprocessing pipelines for real-time predictions
3. **UX Design**: Making data science accessible through game-like interactions
4. **State Management**: Handling session state in Streamlit for score tracking

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select the `streamlit_app.py` file
5. Click "Deploy"

### Other Deployment Options
- Heroku
- AWS EC2
- Google Cloud Platform
- Azure App Service

## 📝 Challenge Submission

This project fulfills the Stack Overflow Code Scientist Challenge requirements:

✅ **Interactive Experience**: Game-based interface for exploring road safety data  
✅ **Deployment Ready**: Can be hosted on Streamlit Cloud or similar platforms  
✅ **Model Integration**: Uses trained XGBoost model for predictions  
✅ **Educational Value**: Helps users understand factors affecting road safety  

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Add new game modes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏆 Acknowledgments

- Kaggle for hosting the Playground Series S5E10 competition
- Stack Overflow for the Code Scientist Challenge
- XGBoost team for the excellent ML library
- Streamlit for making web app development easy

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for the Stack Overflow & Kaggle Code Scientist Challenge**
