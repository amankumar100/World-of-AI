import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load the book ratings data
ratings_data = pd.read_csv('Ratings.csv')

# Split the data into training and test sets
train_data, test_data = train_test_split(ratings_data, test_size=0.2)

# Create a user-item matrix
train_matrix = train_data.pivot_table(index='User_ID', columns='Book_ID', values='Rating')

# Compute item-item similarity using cosine similarity
item_similarity = cosine_similarity(train_matrix.fillna(0))

# Function to get top N similar items for a given item
def get_similar_items(item_id, top_n):
    item_idx = train_data[train_data['Book_ID'] == item_id]['Book_ID'].index[0]
    item_scores = item_similarity[item_idx]
    similar_indices = item_scores.argsort()[::-1][1:top_n+1]
    similar_items = train_data.iloc[similar_indices]['Book_ID'].values
    return similar_items

# Function to make book recommendations for a user
def recommend_books(user_id, top_n):
    user_ratings = train_data[train_data['User_ID'] == user_id]
    recommended_books = []
    for _, row in user_ratings.iterrows():
        similar_items = get_similar_items(row['Book_ID'], top_n)
        recommended_books.extend(similar_items)
    return list(set(recommended_books))

# Test the recommendation system
user_id = 123
top_n = 5
recommended_books = recommend_books(user_id, top_n)
print(f"Top {top_n} recommended books for User {user_id}:")
for book_id in recommended_books:
    print(f"Book ID: {book_id}")

