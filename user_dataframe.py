import pandas as pd  # To read csv file into DataFrame

movies = pd.read_csv('data/ml-latest-small/movies.csv',
                     usecols=["movieId", "title", "genres"])
ratings = pd.read_csv('data/ml-latest-small/ratings.csv',
                      usecols=["userId", "movieId", "rating", "timestamp"])
links = pd.read_csv('data/ml-latest-small/links.csv',
                    usecols=["movieId", "imdbId", "tmdbId"])
tags = pd.read_csv('data/ml-latest-small/tags.csv',
                   usecols=["userId", "movieId", "tag", "timestamp"])

user_movie_df = ratings.pivot(
    index='userId', columns='movieId', values='rating').fillna(0)
# print(user_movie_matrix)


# returns movies as json
def get_movies_json():
    movies_long = pd.merge(links, movies, on='movieId')
    movies_long.to_csv('export_dataframe.csv', index=None, header=True)
    json = movies_long.to_json('export_df.json', orient='records');
    return json