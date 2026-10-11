import argparse
import os
import pickle
from modules.train_functions import train_nmf_model, train_svd1_model, train_svd2_model, train_sgd_model
from modules.predict_functions import predict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and Predict Recommender Models (NMF, SVD1, SVD2, SGD)")
    parser.add_argument("--train", type=str, default="no",
                        help="Train mode: 'yes' to train models, 'no' otherwise.")
    parser.add_argument("--predict", type=str, default="no",
                        help="Predict mode: 'yes' to predict ratings, 'no' otherwise.")
    parser.add_argument("--train_file", type=str, default="data/ratings.csv",
                        help="CSV file with training data (userId,movieId,rating).")
    parser.add_argument("--input_file", type=str, default="data/test_file.csv",
                        help="CSV file with (userId,movieId) for predictions.")
    parser.add_argument("--model_path", type=str, default="results/model.pkl",
                        help="Path to save/load the trained model(s).")
    parser.add_argument("--output_file", type=str, default="results/preds.csv",
                        help="Where to save predictions.")
    parser.add_argument("--alg", type=str, choices=["NMF", "SVD1", "SVD2", "SGD", "ALL"],
                        help="Algorithm to use: NMF, SVD1, SVD2, SGD or ALL.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    train_mode = (args.train.lower() == "yes")
    predict_mode = (args.predict.lower() == "yes")

    models = {}
    if train_mode:
        print(f"Training mode activated with algorithm: {args.alg}")
        if args.alg in ["NMF", "ALL"]:
            Z_approx, user_map, movie_map, mask = train_nmf_model(args.train_file)
            models["NMF"] = {"Z_approx": Z_approx, "user_map": user_map, "movie_map": movie_map}

        if args.alg in ["SVD1", "ALL"]:
            Z_approx, user_map, movie_map, mask = train_svd1_model(args.train_file)
            models["SVD1"] = {"Z_approx": Z_approx, "user_map": user_map, "movie_map": movie_map}

        if args.alg in ["SVD2", "ALL"]:
            Z_approx, user_map, movie_map, mask = train_svd2_model(args.train_file)
            models["SVD2"] = {"Z_approx": Z_approx, "user_map": user_map, "movie_map": movie_map}

        if args.alg in ["SGD", "ALL"]:
            Z_approx, user_map, movie_map, mask = train_sgd_model(args.train_file)
            models["SGD"] = {"Z_approx": Z_approx, "user_map": user_map, "movie_map": movie_map}

        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        with open(args.model_path, "wb") as f:
            pickle.dump(models, f)
        print(f"Model(s) saved to {args.model_path}")

    if predict_mode:
        print(f"Prediction mode activated with algorithm: {args.alg}")
        if not os.path.exists(args.model_path):
            print(f"Model file {args.model_path} not found. Please train models first.")
            return
        with open(args.model_path, "rb") as f:
            models = pickle.load(f)
        
        model = models[args.alg.upper()]
        predictions = predict(args.input_file, model)
        

        # Save predictions
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            f.write("userId,movieId,rating\n")
            for row in predictions:
                f.write(f"{row['userId']},{row['movieId']},{row['rating']}\n")
        print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
