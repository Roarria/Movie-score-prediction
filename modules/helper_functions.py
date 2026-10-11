import pandas as pd
import numpy as np

def build_rating_matrix(train_file):
    """
    Reads a ratings CSV file with columns: userId, movieId, rating.
    Builds and returns the user–movie matrix Z with missing entries filled 
    using the average of the user's mean and the movie's mean rating.
    Also returns mappings from userId to row index and movieId to column index
    and a boolean mask indicating where the original ratings were present.

    Parameters:
      - train_file (str): Path to the training CSV file.

    Returns:
      - Z (ndarray): Filled rating matrix of shape (n_users, n_movies).
      - user_map (dict): Mapping from userId to row index.
      - movie_map (dict): Mapping from movieId to column index.
      - mask (ndarray): Boolean matrix of shape (n_users, n_movies),
                        where True indicates original data, False means imputed.
    """
    df = pd.read_csv(train_file)

    # Extract unique users and movies
    unique_users = df["userId"].unique()
    unique_movies = df["movieId"].unique()

    # Create mappings
    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    n_users = len(user_map)
    n_movies = len(movie_map)

    # Initialize matrix and mask
    Z = np.zeros((n_users, n_movies), dtype=np.float32)
    mask = np.zeros((n_users, n_movies), dtype=bool)

    # Filling known ratings
    for row in df.itertuples():
        u = row.userId
        m = row.movieId
        rating = row.rating
        i = user_map[u]
        j = movie_map[m]
        Z[i, j] = rating
        mask[i, j] = True

    # Compute per-user and per-movie means
    user_means = np.true_divide(Z.sum(axis=1), mask.sum(axis=1), where=mask.sum(axis=1)!=0)
    movie_means = np.true_divide(Z.sum(axis=0), mask.sum(axis=0), where=mask.sum(axis=0)!=0)
    global_mean = np.mean(Z[mask])

    # Fill missing values
    for i in range(n_users):
        for j in range(n_movies):
            if not mask[i, j]:    # all missing values
                has_user_data = mask[i].any()
                has_movie_data = mask[:, j].any()

                if has_user_data and has_movie_data:
                    Z[i, j] = (user_means[i] + movie_means[j]) / 2
                elif has_user_data:
                    Z[i, j] = user_means[i]
                elif has_movie_data:
                    Z[i, j] = movie_means[j]
                else:
                    Z[i, j] = global_mean

    return Z, user_map, movie_map, mask
