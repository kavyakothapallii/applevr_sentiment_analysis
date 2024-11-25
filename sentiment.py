import pandas as pd
from textblob import TextBlob
import os


file_path = 'C:\\Users\\hp\\Desktop\\python\\applevr.xlsx'  

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' does not exist.")
else:
    df = pd.read_excel(file_path, skiprows=1)

    df.columns = ['Comments']

    print("Initial DataFrame:")
    print(df.head())

    def analyze_sentiment(text):
        if pd.isna(text):  
            return 0 
        blob = TextBlob(text)
        return blob.sentiment.polarity

   
    df['sentiment_score'] = df['Comments'].apply(analyze_sentiment)

    
    def categorize_sentiment(score):
        if score > 0:
            return 'Positive'
        elif score < 0:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)

    print("\nDataFrame with Sentiment Scores and Categories:")
    print(df[['Comments', 'sentiment_score', 'sentiment_category']].head())

   
    output_file_path = 'analyzed_applevr_comments.xlsx'
    df.to_excel(output_file_path, index=False)

    print(f"\nResults saved to {output_file_path}")
