import svd
import cf_knn
import pandas as pd
import numpy as np
from user_dataframe import ratings, user_movie_df, movies, get_movies_json

# print("USERS TOP MOVIES:")
# user_ratings = user_movie_matrix.iloc[userId - 1]
# top_user_ratings = user_ratings.sort_values(ascending=False).to_frame()
# print(pd.merge(top_user_ratings, movies, on='movieId').head(10).to_string())

movieIds = [1, 2, 19]
print(svd.recommend(movieIds, print_output=False))
# cf_knn.recommend(userId)

# print(get_movies_json())