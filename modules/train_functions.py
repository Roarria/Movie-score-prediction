import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
import torch
from .helper_functions import build_rating_matrix



def train_nmf_model(train_file, n_components=30):
    """
    Reads the ratings CSV file, builds the rating matrix using build_rating_matrix,
    performs NMF, and returns the approximated rating matrix along with mappings
    and mask placing original ratings.

    Parameters:
      - train_file (str): Path to the training CSV file.
      - n_components (int): Rank for the NMF decomposition.

    Returns:
      - Z_approx (ndarray): Approximated rating matrix from NMF.
      - user_map (dict): Mapping from userId to row index.
      - movie_map (dict): Mapping from movieId to column index.
    """
    Z, user_map, movie_map, mask = build_rating_matrix(train_file)

    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=500)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)

    return Z_approx, user_map, movie_map, mask


def train_svd1_model(train_file, n_components=15):
    """
    Reads the ratings CSV file, builds the rating matrix using build_rating_matrix,
    performs SVD, and returns the approximated rating matrix along with mappings
    and mask placing original ratings.

    Parameters:
      - train_file (str): Path to the training CSV file.
      - n_components (int): Rank for the SVD decomposition.

    Returns:
      - Z_approx (ndarray): Approximated rating matrix from NMF.
      - user_map (dict): Mapping from userId to row index.
      - movie_map (dict): Mapping from movieId to column index.
    """
    Z, user_map, movie_map, mask = build_rating_matrix(train_file)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(Z)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_

    W = svd.transform(Z) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    Z_approx = np.dot(W, H)

    return Z_approx, user_map, movie_map, mask


def train_svd2_model(train_file, n_components=1, max_iter=2000, tol=1e-4, verbose=False):
    Z, user_map, movie_map, mask = build_rating_matrix(train_file)
    Z_copy = Z.copy()         # creating copy to prevent changing Z matrix (in case od alg == ALL)
    
    for iter in range(max_iter):
        # SVD on current approximation
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(Z_copy)

        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_
        W = svd.transform(Z_copy) / svd.singular_values_
        H = np.dot(Sigma2, VT)
        Z_approx = np.dot(W, H)

        # Overwriting known data from original matrix Z
        Z_new = Z_approx.copy()
        Z_new[mask] = Z[mask]

        # Stop criteria:    depends on changes on predicted data
        diff = np.linalg.norm(Z_new[~mask] - Z_copy[~mask])
        if diff < tol:
            if verbose == True:
                print(iter)
            break

        Z_copy = Z_new

    return Z_copy, user_map, movie_map, mask


def train_sgd_model(train_file, n_components=1, lr=0.001, max_iter=2000, optimizer_name="adam", tol=1e-2, verbose=True):
    Z, user_map, movie_map, mask = build_rating_matrix(train_file)
    Z_tensor = torch.tensor(Z, dtype=torch.float32)

    num_users, num_movies = Z.shape

    # Initializing W and H randomly
    torch.manual_seed(12)
    W = torch.rand((num_users, n_components), requires_grad=False, device='cpu') * 5 / np.sqrt(n_components)      # mean = 2.5, distribution N(0,5)
    H = torch.rand((n_components, num_movies), requires_grad=False, device='cpu') * 5 / np.sqrt(n_components)     # mean = 2.5, distribution N(0,5)
    W = W.requires_grad_(True)
    H = H.requires_grad_(True)

    # Choose optimizer
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD([W, H], lr=lr)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam([W, H], lr=lr)
    else:
        raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")
    
    loss_fn = torch.nn.MSELoss()

    prev_pred = None
    for epoch in range(max_iter):
        optimizer.zero_grad()

        Z_pred = torch.matmul(W, H)
        loss = loss_fn(Z_pred[mask], Z_tensor[mask])

        loss.backward()
        optimizer.step()

        # Stop criteria:    depends on changes on Z_pred
        if prev_pred is not None:
            diff = torch.norm(Z_pred - prev_pred).item()
            if diff < tol:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}, Δ = {diff:.6f}")
                break
        prev_pred = Z_pred.clone().detach()

        if verbose==True and ((epoch+1) % 100 == 0 or epoch == 0):
            print(f"Epoch {epoch+1}/{max_iter}, Loss: {loss.item():.4f}")

    Z_approx = (W @ H).detach().numpy()
    return Z_approx, user_map, movie_map, mask    #poprawic?
