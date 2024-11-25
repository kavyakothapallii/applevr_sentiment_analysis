import pandas as pd
from googleapiclient.discovery import build

# Replace with your actual API key and video ID
API_KEY = 'AIzaSyAYri9oKfUPE-D2dL-AZncvufWpiGJjmHo'
VIDEO_ID = '5MhRZp2uunc'

def get_youtube_comments(video_id, api_key, max_results=5000):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100,  # Fetching in batches of 100
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'author': comment['authorDisplayName'],
                'text': comment['textDisplay'],
                'likes': comment['likeCount'],
                'published_at': comment['publishedAt'],
            })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break  # No more pages

    return comments

def main():
    comments = get_youtube_comments(VIDEO_ID, API_KEY, max_results=5000)

    if comments:
        df = pd.DataFrame(comments)
        df.to_csv('youtube_comments.csv', index=False)
        print(f"{len(comments)} comments saved to youtube_comments.csv")
    else:
        print("No comments found or an error occurred.")

if _name_ == "_main_":
    main()
