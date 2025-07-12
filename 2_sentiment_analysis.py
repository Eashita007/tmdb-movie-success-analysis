import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
df = pd.read_csv('Dataset/cleaned_tmdb_data.csv')

sid = SentimentIntensityAnalyzer()

# Apply VADER
df['sentiment_score'] = df['overview'].fillna('').apply(lambda x: sid.polarity_scores(x)['compound'])

# Save updated data
df.to_csv('Dataset/tmdb_with_sentiment.csv', index=False)
