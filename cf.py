from user_dataframe import ratings
from user_dataframe import user_movie_matrix
from user_dataframe import movies
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def recommend(target_user_id, n_recommendations=10, print_output=True):

    # user similarity
    similarity = cosine_similarity(user_movie_matrix)
    sum_similarity = np.absolute(similarity).sum(axis=0)
    pred_df = pd.DataFrame(similarity.dot(user_movie_matrix), columns=user_movie_matrix.columns).divide(sum_similarity,
                                                                                                        axis=0)
    # for a given user, select top recommendations for movies not already rated
    user_index = target_user_id - 1
    user_ratings = user_movie_matrix.iloc[user_index]
    already_rated_labels = list(user_ratings.iloc[user_ratings.nonzero()[0]].index)
    # drop already rated movies and then sort by top recommendations
    results_df = pred_df.iloc[user_index].drop(labels=already_rated_labels).sort_values(ascending=False).to_frame()

    final_recs = pd.merge(results_df, movies, on='movieId')[['movieId', 'title', 'genres']].head(n_recommendations)

    if print_output:
        # print output:
        print("USERS TOP MOVIES:")
        top_user_ratings = user_ratings.sort_values(ascending=False).to_frame()
        print(pd.merge(top_user_ratings, movies, on='movieId').head(15).to_string())

        print('TOP %d RECOMMENDATIONS:' % n_recommendations)
        print(final_recs.to_string())

    # return prediction matrix and final recommendations
    return pred_df, final_recs