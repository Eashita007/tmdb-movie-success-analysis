import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load the processed dataset with sentiment
df = pd.read_csv('Dataset/tmdb_with_sentiment.csv')

# Optional: Drop rows with missing or infinite values (cleanup)
df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()

# Feature engineering
df['num_genres'] = df['genres'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
df['title_word_count'] = df['original_title'].apply(lambda x: len(str(x).split()))
df['cast_count'] = df['cast'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)

# Define features and target
features = df[['budget', 'popularity', 'vote_average', 'vote_count', 'sentiment_score', 'num_genres', 'cast_count']]
target = df['revenue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"🎯 Random Forest R² Score: {r2:.4f}")
print(f"📉 Random Forest MSE: {mse:.2e}")
