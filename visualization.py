import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path = '/content/analyzed_applevr_comments.xlsx'
data = pd.read_excel(file_path)
score_stats = data['sentiment_score'].describe()
category_counts = data['sentiment_category'].value_counts()
mean_score_by_category = data.groupby('sentiment_category')['sentiment_score'].mean()
overall_sentiment = 'Positive' if mean_score_by_category['Positive'] > mean_score_by_category['Negative'] else 'Negative'
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#99ff99','#ff6666'], ax=ax[0])
ax[0].set_title('Sentiment Category Distribution')
ax[0].set_ylabel('')
summary_text = (
    f"Total Comments: {int(score_stats['count'])}\n"
    f"Mean Sentiment Score: {score_stats['mean']:.6f}\n"
    f"Positive Comments: {category_counts['Positive']} ({category_counts['Positive'] / score_stats['count']:.1%})\n"
    f"Neutral Comments: {category_counts['Neutral']} ({category_counts['Neutral'] / score_stats['count']:.1%})\n"
    f"Negative Comments: {category_counts['Negative']} ({category_counts['Negative'] / score_stats['count']:.1%})\n\n"
    f"Mean Sentiment Score by Category:\n"
    f"Positive: {mean_score_by_category['Positive']:.6f}\n"
    f"Neutral: {mean_score_by_category['Neutral']:.6f}\n"
    f"Negative: {mean_score_by_category['Negative']:.6f}\n\n"
    f"Overall, the sentiment towards Apple Vision Pro is predominantly {overall_sentiment}."
)
ax[1].axis('off')
ax[1].text(0, 0.5, summary_text, ha='left', va='center', fontsize=12, wrap=True)

plt.show()
