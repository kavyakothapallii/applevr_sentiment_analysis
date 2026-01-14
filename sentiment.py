import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

df = pd.read_excel("applevr_comments.xlsx")

model_name = "yangheng/deberta-v3-base-absa-v1.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

target_aspect = "Apple Vision Pro"

def analyze_aspect_sentiment(text, aspect):

    result = classifier(text, text_pair=aspect, truncation=True)
    best = max(result[0], key=lambda x: x['score'])
    return best['label'], best['score']

labels = []
scores = []
for txt in df['Comments'].astype(str):
    label, score = analyze_aspect_sentiment(txt, target_aspect)
    labels.append(label)
    scores.append(score)

df['sentiment_label'] = labels
df['sentiment_score'] = scores

label_to_num = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df['sentiment_numeric'] = df['sentiment_label'].map(label_to_num)

df.to_csv("applevr_comments_with_aspect_sentiment.csv", index=False)
print(df.head())
