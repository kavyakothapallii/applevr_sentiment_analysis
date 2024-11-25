import praw
import pandas as pd
CLIENT_ID = 'Enter your client ID'
CLIENT_SECRET = 'Enter your client secret Key'
USER_AGENT = 'python:apple_vision_pro_script:v1.0 (by /u/Bright_Chemical697)'
USERNAME = 'Enter your username'
PASSWORD = 'Enter account password'
def is_review(submission):
    review_keywords = ['review', 'experience', 'thoughts', 'opinion', 'feedback']
    content = (submission.title + ' ' + submission.selftext).lower()
    return any(keyword in content for keyword in review_keywords)
def get_reddit_reviews(subreddit_name, search_query, desired_limit=1000):
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        username=USERNAME,
        password=PASSWORD
    )
    subreddit = reddit.subreddit(subreddit_name)
    reviews = []
    after = None
    while len(reviews) < desired_limit:
        submissions = subreddit.search(search_query, limit=100, params={'after': after})
        count = 0
        for submission in submissions:
            if len(reviews) >= desired_limit:
                break
            if is_review(submission):
                reviews.append({
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'score': submission.score,
                    'url': submission.url,
                    'num_comments': submission.num_comments
                })
            count += 1
            after = submission.name
        if count == 0:
            break
    return reviews
def main():
    subreddit_name = 'technology'
    search_query = 'Apple Vision Pro review'
    desired_limit = 1000  
    reviews = get_reddit_reviews(subreddit_name, search_query, desired_limit)
    if reviews:
        df = pd.DataFrame(reviews)
        df.to_csv('reddit_apple_vision_pro_reviews.csv', index=False)
        print(f"{len(reviews)} reviews saved to reddit_apple_vision_pro_reviews.csv")
    else:
        print("No reviews found or an error occurred.")
if _name_ == "_main_":
    main()
