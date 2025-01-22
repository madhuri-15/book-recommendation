# Import libraries

import nltk
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template


# Read dataset
book_df = pd.read_csv("./data/final_book_df.csv", encoding='latin-1')
filter_data = pd.read_csv("./data/filter_data.csv")

# Round ratings to 1 decimal point
book_df['weighted_rating'] = book_df['weighted_rating'].apply(lambda x: round(x, 1))

# Get the list of stopwords in english
stop_words = set(stopwords.words('english'))

"""Find similar book based on search text keyword."""
def search_book(search_text, data=filter_data, n_recommendations=10):
    # Tokenize the text
    keywords = word_tokenize(search_text.lower())

    # Filter out the stopwords
    filtered_kw = [word for word in keywords if word not in stop_words]
    
    # Get the unique book title from the filtered data b.
    book_titles = data.book_title.unique().tolist()

    # Search the similar books for the given search text.
    similar_books = []
    for book in book_titles:
        if any(word in book for word in filtered_kw):
            similar_books.append(book)
            
    if similar_books:
        result = book_df[book_df.book_title.isin(similar_books)].head(n_recommendations)
        return result.T.to_dict()
    else:
        return None


def get_data_by(age):
    if age:
        if age < 13:
            return filter_data[filter_data.age_grp == 'Childern']
        elif 12 < age <= 18:
            return filter_data[filter_data.age_grp == 'Teens']
        elif 18 < age <= 35:
            return filter_data[filter_data.age_grp == 'Young Adults']
        elif 35 < age <= 60:
            return filter_data[filter_data.age_grp == 'Adults']
        elif age > 60:
            return filter_data[filter_data.age_grp == 'Seniors']
    else:
        return filter_data
    

"""Book recommendation system using Cosine Similarity."""
def recommendation_system(search_text, n_recommendations=10, age=None):
    search_text = search_text.lower()
    # Filter data by age
    filtered_book_df = get_data_by(age)

    try:
        # Create pivot table that displays the users with ratings
        rating_pt = pd.pivot_table(filtered_book_df, index='isbn', columns='user_id', values='book_rating')
        rating_pt = rating_pt.fillna(0)
        
        # Compute the similarity score
        similarity_score = cosine_similarity(rating_pt)
        cosimilarity_matrix = pd.DataFrame(similarity_score, columns=rating_pt.index, index=rating_pt.index)
        
        # Search book title in cosimilarity matrix.
        search_book_isbn = book_df[book_df.book_title == search_text]['isbn'].values[0]
        similar_books = cosimilarity_matrix[search_book_isbn].sort_values(ascending=False)[:n_recommendations].index
        similar_books_df = pd.DataFrame({'isbn': similar_books})
        result = pd.merge(similar_books_df, book_df, on='isbn', how='left')
        return result.T.to_dict()
        
    except:
        return search_book(search_text, filtered_book_df, n_recommendations)


# Application

app = Flask(__name__)

@app.route("/")
def index():
    books = book_df.sort_values(by=['weighted_rating'], ascending=False).head(50).T.to_dict()
    return render_template('index.html', books=books)


@app.route("/advanced-search", methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_book = request.form.get('search_text')
        age = request.form.get('age')
        if age:
            age = int(age)
        books = recommendation_system(search_book, n_recommendations=10, age=age)
        return render_template('search_books.html', books=books)
    
    return render_template('search_books.html')



if "__main__" == __name__:
    app.run(host="localhost", port=5000, debug=True)