import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold

###############################################################################################

def calculate_cv_scores(all_user_ratings, n_splits, top_k, all_books, all_user_toread, recommendation_sys):
    """
    Function to calculate kfold cross-validated precision@k, recall@k, mean average precision@k and ndcg.
    
    Parameters:
    - all_user_ratings: Table of user ratings for books
    - n_splits: Number of kfold splits
    - top_k: Number of top recommendations to consider
    - all_books: Table of book metadata
    - all_user_toread: To read list for all users
    - recommendation_sys: Chosen recommendation system (e.g. popularity_rec) 
    
    Returns:
    - precision_at_k: A dictionary containing Precision@K scores for each fold
    - recall_at_k: A dictionary containing Recall@k scores for each fold
    - map_at_k: A dictionary containing Mean average precision @ k scores for each fold
    """
    
    # Output dictionaries
    precision_at_k = {}
    recall_at_k = {}
    map_at_k = {}
    
    kf = KFold(n_splits, shuffle=True, random_state=0)
    for n, (train_index, valid_index) in enumerate(kf.split(all_user_ratings)):

        # Create training/validation sets
        train_data = all_user_ratings.iloc[train_index]
        valid_data = all_user_ratings.iloc[valid_index]

        # Calculate average evaluation metric scores across users for kth fold
        precision_at_k[n], recall_at_k[n], map_at_k[n] = calculate_avg_scores(train_data, valid_data, top_k, all_books, all_user_toread, recommendation_sys)


    # Calculate average evaluation metric scores across k folds
    avg_precision_at_k = sum(precision_at_k.values()) / len(precision_at_k)
    avg_recall_at_k = sum(recall_at_k.values()) / len(recall_at_k)
    avg_map_at_k = sum(map_at_k.values()) / len(map_at_k)
        
    return avg_precision_at_k, avg_recall_at_k, avg_map_at_k

###############################################################################################

def calculate_avg_scores(train_data, valid_data, top_k, all_books, all_user_toread, recommendation_sys):
    """
    Function to calculate precision, recall, mean average precision and ndcg given training/validation data.

    Parameters:
    - train_data: List of user ratings for each book
    - valid_data: List of user ratings for each book
    - top_k: Number of top recommendations to consider
    - all_books: Table of book metadata
    - all_user_toread: To read list for all users
    - recommendation_sys: Chosen recommendation system (e.g. popularity_rec) 

    Returns:
    - overall_precision: Average precision@K across users in the validation data
    - overall_recall: Average Recall@k across users in the validation data
    - overall_map: Average Mean average precision @ k across users in the validation data
    """
    user_recommendations = {}
    user_history = {}
    user_precision = {}
    user_recall = {}
    user_map = {}

    # Randomly sample user ids from the validation set
    user_ids = list(valid_data["user_id"].unique())
    sampled_user_ids = random.sample(user_ids, int(0.1*len(user_ids)))

    for id in sampled_user_ids:
        
        # Output a dictionary containing 50 predicted items for each user.
        user_recommendations[id] = recommendation_sys(id, train_data, 50, all_books, all_user_toread)
    
        # Output a dictionary containing actual rated items for each user.
        user_history[id] = valid_data.loc[valid_data["user_id"] == id, "book_id"].tolist()

        # Get lists of predicted & actual items for the user, default to an empty list
        predicted = user_recommendations.get(id, [])
        actual = user_history.get(id, [])
    
        # Output a dictionary containing metric scores for each user 
        user_precision[id] = calculate_precision_at_k(actual, predicted, top_k)
        user_recall[id] = calculate_recall_at_k(actual, predicted, top_k)
        user_map[id] = calculate_mean_average_precision_at_k(actual, predicted, top_k)
        
    # Average metric over all users
    overall_precision = sum(user_precision.values()) / len(user_precision)
    overall_recall = sum(user_recall.values()) / len(user_recall)
    overall_map = sum(user_map.values()) / len(user_map)

    return overall_precision, overall_recall, overall_map

###############################################################################################

def calculate_precision_at_k(actual, predicted, top_k):
    """
    Function to calculate Precision@K for a given a user.

    Parameters:
    - actual: List of actual relevant items
    - predicted: List of predicted items
    - top_k: Number of top recommendations to consider

    Returns:
    - precision_at_k: Precision@K score
    """
    
    # Consider only the top-K predicted items
    predicted_at_k = predicted[:top_k]

    # Calculate the number of common items between actual and predicted
    true_positives = len(set(actual).intersection(set(predicted_at_k)))

    # Calculate Precision@K
    precision_at_k = true_positives / top_k if top_k > 0 else 0.0

    return precision_at_k

###############################################################################################

def calculate_recall_at_k(actual, predicted, top_k):
    """
    Function to calculate Recall@K for a given a user.

    Parameters:
    - actual: List of actual relevant items
    - predicted: List of predicted items
    - top_k: Number of top recommendations to consider

    Returns:
    - recall_at_k: Recall@K score
    """
    # Consider only the top-K predicted items
    predicted_at_k = predicted[:top_k]

    # Calculate the number of common items between actual and predicted
    true_positives = len(set(actual).intersection(set(predicted_at_k)))

    # Calculate the total number of relevant items
    total_relevant_items = len(actual)

    # Calculate Recall@K
    recall_at_k = true_positives / total_relevant_items if total_relevant_items > 0 else 0.0

    return recall_at_k

###############################################################################################

def calculate_mean_average_precision_at_k(actual, predicted, top_k):
    """
    Function to calculate mean average precision@K for a given a user.

    Parameters:
    - actual: List of actual relevant items
    - predicted: List of predicted items
    - top_k: Number of top recommendations to consider

    Returns:
    - mean average_precision_at_k: Mean Average Precision@K score
    """
    precision_at_k_values = []

    # Calculate precision at each position up to K
    for i in range(1, top_k + 1):
        precision_at_i = calculate_precision_at_k(actual, predicted, i)
        precision_at_k_values.append(precision_at_i)

    # Calculate Average Precision@K
    mean_average_precision_at_k = sum(precision_at_k_values) / top_k if top_k > 0 else 0.0

    return mean_average_precision_at_k

###############################################################################################

def calculate_ndcg(relevance_scores):
    """
    Function to calculate Normalized Discounted Cumulative Gain (NDCG) for a given a user.

    Parameters:
    - relevance_scores: List of relevance scores for recommended items

    Returns:
    - ndcg: NDCG score
    """
    # Sort the relevance scores in descending order
    sorted_scores = sorted(relevance_scores, reverse=True)

    # Calculate DCG
    dcg = sum((2**score - 1) / (math.log2(i + 2)) for i, score in enumerate(sorted_scores))

    # Calculate Ideal DCG (IDCG)
    ideal_sorted_scores = sorted([2, 1] + [0] * (len(relevance_scores) - 2), reverse=True)
    idcg = sum((2**score - 1) / (math.log2(i + 2)) for i, score in enumerate(ideal_sorted_scores))

    # Calculate NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg

###############################################################################################