def run(df):
    # Use lowercase column names if uppercase not present
    likes_col = 'Likes' if 'Likes' in df.columns else 'likes'
    comments_col = 'Comments' if 'Comments' in df.columns else 'comments_count'
    post_id_col = 'Post ID' if 'Post ID' in df.columns else 'post_id'
    caption_col = 'Caption' if 'Caption' in df.columns else 'caption'
    top_posts = df.sort_values(by=[likes_col, comments_col], ascending=False).head(5)
    print("Top 5 Recommended Posts:")
    print(top_posts[[post_id_col, caption_col, likes_col, comments_col]])
