import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Dataset/tmdb_with_sentiment.csv')

# Sentiment by Genre (Boxplot)
plt.figure(figsize=(12,6))
sns.boxplot(x='main_genre', y='sentiment_score', data=df)
plt.xticks(rotation=45)
plt.title("Sentiment Score by Genre")
plt.tight_layout()
plt.savefig('outputs/sentiment_genre_plot.png')

# Budget vs Revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x='budget', y='revenue', data=df)
plt.title("Budget vs Revenue")
plt.savefig('outputs/budget_vs_revenue.png')

# Average sentiment per genre
avg_sentiment = df.groupby('main_genre')['sentiment_score'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
avg_sentiment.plot(kind='bar')
plt.title("Average Sentiment per Genre")
plt.ylabel("Sentiment Score")
plt.tight_layout()
plt.savefig('outputs/genre_sentiment_bar.png')
