import pandas as pd
import openai

OPENAI_API_KEY = 'API-KEY'
input_csv = 'Input/Path.csv'
output_csv = 'output/Path.csv'

openai.api_key = OPENAI_API_KEY

def categorize_comments_with_chatgpt(comments):
    categorized_comments = []
    
    # Define the system message
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that categorizes comments based on different views like angry, happy, excited, etc. Please review each comment and return the appropriate category."
    }
    
    # Iterate through comments
    for comment in comments:
        messages = [
            system_message,
            {"role": "user", "content": comment['text']}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=10
        )
        
        category = response.choices[0].message['content'].strip().lower()
        comment['category'] = category
        categorized_comments.append(comment)
    
    return categorized_comments

def main():
    df = pd.read_csv(input_csv)
    comments = df.to_dict('records')
    
    categorized_comments = categorize_comments_with_chatgpt(comments)
    
    if categorized_comments:
        categorized_df = pd.DataFrame(categorized_comments)
        categorized_df.to_csv(output_csv, index=False)
        print(f"Categorized comments saved to {output_csv}")
    else:
        print("No comments categorized or an error occurred.")

if _name_ == "_main_":
    main()
