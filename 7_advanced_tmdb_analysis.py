# This file will contain code for all advanced enhancements
# Split into code blocks for each suggested feature

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# Load dataset
movies = pd.read_csv("Dataset/tmdb_with_sentiment.csv")
movies = movies.replace([float('inf'), float('-inf')], pd.NA).dropna()

# Basic feature engineering
movies['num_genres'] = movies['genres'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
movies['cast_count'] = movies['cast'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)

# Define features and target
features = movies[['budget', 'popularity', 'vote_average', 'vote_count', 'sentiment_score', 'num_genres', 'cast_count']]
target = movies['revenue']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# -----------------------------
# GridSearchCV for Random Forest
# -----------------------------
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n🔧 Best Parameters (Random Forest):", grid_search.best_params_)
print("📈 Best R2 Score (Cross-Validated):", round(grid_search.best_score_, 4))

# Save the best model
import joblib
joblib.dump(grid_search.best_estimator_, "models/best_random_forest_model.pkl")
print("✅ Best Random Forest model exported to: models/best_random_forest_model.pkl")

# -----------------------------
# Feature Importance Visualization
# -----------------------------
best_model = grid_search.best_estimator_
importances = best_model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(features.columns, importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
print("✅ Saved to: outputs/feature_importance.png")

# -----------------------------
# XGBoost Model
# -----------------------------
try:
    import xgboost as xgb

    print("\n⚡ Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    xgb_r2 = r2_score(y_test, y_pred_xgb)
    xgb_mse = mean_squared_error(y_test, y_pred_xgb)

    print("XGBoost R² Score:", round(xgb_r2, 4))
    print("XGBoost MSE:", format(xgb_mse, ".2e"))
except ModuleNotFoundError:
    print("\n⚠️ Skipped XGBoost — module not installed. Run: pip install xgboost")
    xgb_r2 = None

# -----------------------------
# KMeans Clustering
# -----------------------------
cluster_features = movies[['sentiment_score', 'budget', 'revenue']]
kmeans = KMeans(n_clusters=4, random_state=42)
movies['cluster'] = kmeans.fit_predict(cluster_features)
movies.to_csv("Dataset/movies_with_clusters.csv", index=False)
print("✅ Cluster labels saved to: Dataset/movies_with_clusters.csv")

# -----------------------------
# Model Comparison Plot
# -----------------------------
print("\n📊 Generating model comparison plot...")
linear_r2 = 0.6561  # from earlier run
rf_r2 = grid_search.best_score_

if xgb_r2:
    models = ['Linear Regression', 'Random Forest', 'XGBoost']
    scores = [linear_r2, rf_r2, xgb_r2]
else:
    models = ['Linear Regression', 'Random Forest']
    scores = [linear_r2, rf_r2]

plt.figure(figsize=(6, 4))
plt.bar(models, scores, color=['skyblue', 'seagreen', 'orange'][:len(models)])
plt.ylabel('R² Score')
plt.title('Model Performance Comparison')
plt.ylim(0.6, 0.75)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('outputs/model_comparison.png')
print("✅ Saved model comparison plot to: outputs/model_comparison.png")
