import pandas as pd
import ast

# Use the correct relative path
movies = pd.read_csv('Dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('Dataset/tmdb_5000_credits.csv')

# Merge on movie ID
df = movies.merge(credits, left_on='id', right_on='movie_id')

# Convert genres from string to list
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
df['main_genre'] = df['genres'].apply(lambda x: x[0] if x else None)

# Keep rows with valid budget and revenue
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

# Save cleaned data
df.to_csv('Dataset/cleaned_tmdb_data.csv', index=False)
