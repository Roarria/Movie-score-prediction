import pandas as pd
import numpy as np


def predict(test_file, model_data):
    """
    Reads a test CSV with columns: userId, movieId.
    Uses the stored Z_approx, user_map, movie_map to produce predictions.
    Missing userId/movieId combos produce a default rating (e.g. average rating).

    Returns a list of dicts with keys: 'userId', 'movieId', 'rating'.
    """
    df = pd.read_csv(test_file)

    Z_approx = model_data["Z_approx"]
    user_map = model_data["user_map"]
    movie_map = model_data["movie_map"]

    predictions = []
    for row in df.itertuples():
        u = row.userId
        m = row.movieId

        if u in user_map and m in movie_map:
            i = user_map[u]
            j = movie_map[m]
            rating = Z_approx[i, j]
        # if movie not seen in training: default to average user rating
        elif u in user_map:
            rating = np.mean(Z_approx[user_map[u],:])
        # if user not seen in training: default to average movie rating
        elif m in movie_map:
            rating = np.mean(Z_approx[:, movie_map[m]])
        # if user and movie not seen in training: default to global average rating (mean from all movies)
        else:
            rating = np.mean(Z_approx)

        # Round rating to nearest 0.5 increment
        rating_rounded = round(rating * 2) / 2

        predictions.append({
            "userId": u,
            "movieId": m,
            "rating": rating_rounded
        })
    return predictions
