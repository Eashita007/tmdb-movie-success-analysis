import streamlit as st
import joblib
import numpy as np

# Page config (for browser tab and icon)
st.set_page_config(page_title="Movie Revenue Predictor", page_icon="🎬")

# Load the trained Random Forest model
model = joblib.load("models/best_random_forest_model.pkl")

# Title and intro
st.title("🎬 TMDB Movie Revenue Predictor")

st.markdown("""
Welcome to the **Movie Revenue Prediction App**!  
This tool predicts a movie's **box office revenue** using machine learning models trained on over 5000 movies from TMDB.

### 🧠 Powered by:
- 🎥 Budget, Popularity, Vote Stats
- 💬 Sentiment Analysis (VADER)
- 🤖 Random Forest with GridSearchCV
- 📊 Feature Engineering & Clustering

🧪 **Try adjusting the values below and click 'Predict Revenue' to see your results!**
""")

# Input sliders
budget = st.slider("Budget ($)", 1_000_000, 300_000_000, step=1_000_000, value=50_000_000)
popularity = st.slider("Popularity Score", 0.0, 300.0, step=1.0, value=50.0)
vote_average = st.slider("Vote Average", 0.0, 10.0, step=0.1, value=7.0)
vote_count = st.slider("Vote Count", 0, 5000, step=100, value=500)
sentiment_score = st.slider("Sentiment Score", -1.0, 1.0, step=0.01, value=0.2)
num_genres = st.slider("Number of Genres", 1, 5, value=2)
cast_count = st.slider("Number of Cast Members", 1, 50, value=10)

# Prediction
if st.button("Predict Revenue"):
    input_data = np.array([[budget, popularity, vote_average, vote_count, sentiment_score, num_genres, cast_count]])
    predicted_revenue = model.predict(input_data)[0]
    st.success(f"🎥 Predicted Box Office Revenue: **${int(predicted_revenue):,}**")
