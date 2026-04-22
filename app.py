import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv')
movies['genres'] = movies['genres'].str.replace('|', ' ')

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['genres']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    result = []
    for i in movie_list:
        result.append(movies.iloc[i[0]].title)

    return result

st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write(movie)