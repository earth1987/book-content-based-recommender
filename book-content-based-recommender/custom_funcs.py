from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import random
import re

###############################################################################################

def pad_embedding(embedding, max_embedding_length):
    """ 
    Function used to pad input arrays
    
    Parameters:
    - embedding: Input embedding
    - max_embedding_length: Maximum embedding length

    Returns: Padded embedding
    """
    if isinstance(embedding, np.ndarray) and len(embedding) > 0:
        return np.pad(embedding, (0, max_embedding_length - len(embedding)), 'constant')
    else:
        return np.zeros(max_embedding_length)

###############################################################################################

def preprocess_text(text, stopwords, lemmatizer):
    """ 
    Function to process text
    
    Parameters:
    - text: Input string
    - stopwords: Set of stopwords to remove
    - lemmatizer: Instantiated lemmatizer object

    Returns: Processed string.
    """
    text = str(text)
    text = text.lower() # Convert to lowercase
    text = re.sub('[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = nltk.word_tokenize(text) # Tokenize
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords] # Lemmatize and remove stopwords
    return " ".join(tokens)

###############################################################################################

def get_word2vec_embedding(text, pretrained_word2vec_model):
    """
    Function to get average word2vec embedding for a given piece of text.

    Parameters:
    - text: Input string
    - pretrained_word2vec_model: Pre-trained word2vec model

    Returns:
    - Average embedding
    """
    tokens = nltk.word_tokenize(text)
    vectors = [pretrained_word2vec_model[token] for token in tokens if token in pretrained_word2vec_model]
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return None

###############################################################################################

def calculate_baseline_cv_score(recommendation_sys, all_user_ratings, all_user_toread, all_books, n_sample, n_splits, top_k):
    """
    Function to calculate kfold cross-validated precision@k, recall@k and mean average precision@k
    
    Parameters:
    - recommendation_sys: Chosen recommendation system (e.g. popularity_rec) 
    - all_user_ratings: DataFrame containing ratings history for all users.
    - all_user_toread: DataFrame containing to read list for all users.
    - all_books: DataFrame containing metadata for all books.
    - n_sample: Number of recommendations to generate.
    - n_splits: Number of kfold splits
    - top_k: Number of top recommendations to consider

    Returns:
    - avg_precision_at_k: Average Precision@K across kfolds
    - avg_recall_at_k: Average Recall@k across kfolds
    - avg_map_at_k: Average Mean average precision @ k across kfolds
    """
    precision_at_k = {}
    recall_at_k = {}
    map_at_k = {}

    # Iterate over kfolds
    kf = KFold(n_splits, shuffle=True, random_state=0)
    for n, (train_index, valid_index) in enumerate(kf.split(all_user_ratings)):

        # Create training/validation sets
        train_data = all_user_ratings.iloc[train_index]
        valid_data = all_user_ratings.iloc[valid_index]

        # Filter new users out of validation set
        unique_train_users = train_data["user_id"].unique()
        unique_train_users = set(unique_train_users)
        valid_data = valid_data.loc[valid_data["user_id"].isin(
            unique_train_users)]

        # Compute evaluation metrics and update relevant dictionaries with scores
        precision_at_k[n], recall_at_k[n], map_at_k[n] = calculate_baseline_kfold_score(recommendation_sys, train_data, valid_data, all_user_toread, all_books, n_sample, top_k)


    # Compute average evaluation metric scores
    avg_precision_at_k = sum(precision_at_k.values()) / len(precision_at_k)
    avg_recall_at_k = sum(recall_at_k.values()) / len(recall_at_k)
    avg_map_at_k = sum(map_at_k.values()) / len(map_at_k)
        
    return avg_precision_at_k, avg_recall_at_k, avg_map_at_k

###############################################################################################

def calculate_baseline_kfold_score(recommendation_sys, train_data, valid_data, all_user_toread, all_books, n_sample, top_k):
    """
    Function to calculate precision, recall and mean average precision given training/validation data.

    Parameters:
    - recommendation_sys: Chosen recommendation system (e.g. popularity_rec)
    - train_data: List of user ratings for each book
    - valid_data: List of user ratings for each book
    - all_user_toread: DataFrame containing to read list for all users.
    - all_books: DataFrame containing metadata for all books.
    - n_sample: Number of recommendations to generate.
    - top_k: Number of top recommendations to consider
    
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

    # Iterate over random sample of users from the validation set
    valid_user_ids = valid_data["user_id"].unique()
    valid_user_ids = np.random.choice(valid_user_ids, size=int(
        len(valid_user_ids)*0.1), replace=False)
    for user_id in valid_user_ids:
        
        # Update dictionary with n_sample predicted items for the user
        user_recommendations[user_id] = recommendation_sys(user_id, train_data, all_user_toread, all_books, n_sample)
    
        # Update dictionary with actual rated items for the user
        user_history[user_id] = valid_data.loc[valid_data["user_id"] == user_id, "book_id"].tolist()

        # Compute evaluation metrics and update relevant dictionaries with scores
        predicted = user_recommendations.get(user_id, [])
        actual = user_history.get(user_id, [])
        user_precision[user_id] = calculate_precision_at_k(actual, predicted, top_k)
        user_recall[user_id] = calculate_recall_at_k(actual, predicted, top_k)
        user_map[user_id] = calculate_mean_average_precision_at_k(actual, predicted, top_k)
        
    # Compute average evaluation metric scores
    overall_precision = sum(user_precision.values()) / len(user_precision)
    overall_recall = sum(user_recall.values()) / len(user_recall)
    overall_map = sum(user_map.values()) / len(user_map)

    return overall_precision, overall_recall, overall_map

###############################################################################################

def calculate_cv_score(recommendation_sys, book_matrix, all_user_ratings, all_user_toread, all_books, stopwords, lemmatizer, vectorizer, n_sample, n_splits, top_k):
    """
    Function to calculate kfold cross-validated precision@k, recall@k and mean average precision@k.

    Parameters:
    - recommendation_sys: Chosen recommendation system (e.g. popularity_rec) 
    - book_matrix: Matrix of book vectors
    - all_user_ratings: DataFrame containing ratings history for all users.
    - all_user_toread: DataFrame containing to read list for all users.
    - all_books: DataFrame containing metadata for all books.
    - stopwords: Set of stop words
    - lemmatizer: Instantiated lemmatizer
    - n_sample: Number of recommendations to generate.
    - n_splits: Number of kfold splits
    - top_k: Number of top recommendations to consider

    Returns:
    - avg_precision_at_k: Average Precision@K across kfolds
    - avg_recall_at_k: Average Recall@k across kfolds
    - avg_map_at_k: Average Mean average precision @ k across kfolds
    """

    precision_at_k = {}
    recall_at_k = {}
    map_at_k = {}

    # Iterate over kfolds
    kf = KFold(n_splits, shuffle=True, random_state=0)
    for n, (train_index, valid_index) in enumerate(kf.split(all_user_ratings)):

        # Create training/validation sets
        train_data = all_user_ratings.iloc[train_index]
        valid_data = all_user_ratings.iloc[valid_index]

        # Filter new users out of validation set
        unique_train_users = train_data["user_id"].unique()
        unique_train_users = set(unique_train_users)
        valid_data = valid_data.loc[valid_data["user_id"].isin(
            unique_train_users)]

        # Vectorise user profiles
        all_user_profiles = train_data.groupby("user_id")["book_profile"].agg(
            lambda x: " ".join(x)).rename("user_profile")
        all_user_profiles = all_user_profiles.apply(
            lambda x: preprocess_text(x, stopwords, lemmatizer))
        user_matrix = vectorizer.transform(all_user_profiles.to_numpy())
        user_matrix_ids = all_user_profiles.index.to_numpy()

        # Compute similarity matrix
        cosine_similarity_matrix = cosine_similarity(user_matrix, book_matrix)

        # Compute evaluation metrics and update relevant dictionaries with scores
        precision_at_k[n], recall_at_k[n], map_at_k[n] = calculate_kfold_score(
            recommendation_sys, cosine_similarity_matrix, user_matrix_ids, train_data, valid_data, all_user_toread, all_books, n_sample, top_k)

    # Compute average evaluation metric scores
    avg_precision_at_k = sum(precision_at_k.values()) / len(precision_at_k)
    avg_recall_at_k = sum(recall_at_k.values()) / len(recall_at_k)
    avg_map_at_k = sum(map_at_k.values()) / len(map_at_k)

    return avg_precision_at_k, avg_recall_at_k, avg_map_at_k

###############################################################################################
    
def calculate_kfold_score(recommendation_sys, cosine_similarity_matrix, user_matrix_ids, train_data, valid_data, all_user_toread, all_books, n_sample, top_k):
    """
    Function to calculate precision, recall and mean average precision given training/validation data.

    Parameters:
    - recommendation_sys: Chosen recommendation system (e.g. popularity_rec)
    - cosine_similarity_matrix: Each row corresponds to a user and each column corresponds to a book
    - user_matrix_ids: Numpy array containing user_id corresponding to each row in the similarity matrix
    - train_data: List of user ratings for each book
    - valid_data: List of user ratings for each book
    - all_user_toread: DataFrame containing to read list for all users.
    - all_books: DataFrame containing metadata for all books.
    - n_sample: Number of recommendations to generate.
    - top_k: Number of top recommendations to consider

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

    # Iterate over random sample of users from the validation set
    valid_user_ids = valid_data["user_id"].unique()
    valid_user_ids = np.random.choice(valid_user_ids, size=int(
        len(valid_user_ids)*0.1), replace=False)
    for user_id in valid_user_ids:

        # Update dictionary with n_sample predicted items for the user
        user_recommendations[user_id] = recommendation_sys(
            user_id, cosine_similarity_matrix, user_matrix_ids, train_data, all_user_toread, all_books, n_sample)

        # Update dictionary with actual rated items for the user
        user_history[user_id] = valid_data.loc[valid_data["user_id"]
                                               == id, "book_id"].tolist()

        # Compute evaluation metrics and update relevant dictionaries with scores
        predicted = user_recommendations.get(user_id, [])
        actual = user_history.get(user_id, [])
        user_precision[id] = calculate_precision_at_k(actual, predicted, top_k)
        user_recall[id] = calculate_recall_at_k(actual, predicted, top_k)
        user_map[id] = calculate_mean_average_precision_at_k(
            actual, predicted, top_k)

    # Compute average evaluation metric scores
    overall_precision = sum(user_precision.values()) / len(user_precision)
    overall_recall = sum(user_recall.values()) / len(user_recall)
    overall_map = sum(user_map.values()) / len(user_map)

    return overall_precision, overall_recall, overall_map

###############################################################################################

# def calculate_cv_scores(recommendation_sys, book_matrix, all_user_ratings, all_user_toread, all_books, n_sample, n_splits, top_k):
#     """
#     Function to calculate kfold cross-validated precision@k, recall@k and mean average precision@k.
    
#     Parameters:
#     - recommendation_sys: Chosen recommendation system (e.g. popularity_rec) 
#     - book_matrix: Matrix of book vectors
#     - all_user_ratings: DataFrame containing ratings history for all users.
#     - all_user_toread: DataFrame containing to read list for all users.
#     - all_books: DataFrame containing metadata for all books.
#     - n_sample: Number of recommendations to generate.
#     - n_splits: Number of kfold splits
#     - top_k: Number of top recommendations to consider

#     Returns:
#     - avg_precision_at_k: Average Precision@K across kfolds
#     - avg_recall_at_k: Average Recall@k across kfolds
#     - avg_map_at_k: Average Mean average precision @ k across kfolds
#     """
    
#     # Output dictionaries
#     precision_at_k = {}
#     recall_at_k = {}
#     map_at_k = {}
    
#     kf = KFold(n_splits, shuffle=True, random_state=0)
#     for n, (train_index, valid_index) in enumerate(kf.split(all_user_ratings)):

#         # Create training/validation sets
#         train_data = all_user_ratings.iloc[train_index]
#         valid_data = all_user_ratings.iloc[valid_index]

#         # Create user vectors
#         all_user_profiles = train_data.groupby("user_id")["book_profile"].agg(lambda x: " ".join(x)).rename("user_profile")
#         all_user_profiles = all_user_profiles.apply(lambda x: preprocess_text(x, stopwords, lemmatizer))
#         all_user_vectors = tfidf_vectorizer.transform(all_user_profiles.tolist())

#         # Compute similarity matrix
#         cosine_similarity_matrix = cosine_similarity(all_user_vectors, book_matrix)
                
#         # Calculate average evaluation metric scores across users for kth fold
#         precision_at_k[n], recall_at_k[n], map_at_k[n] = calculate_kfold_score(recommendation_sys, cosine_similarity_matrix, train_data, valid_data, all_user_toread, all_books, n_sample, top_k)


#     # Calculate average evaluation metric scores across k folds
#     avg_precision_at_k = sum(precision_at_k.values()) / len(precision_at_k)
#     avg_recall_at_k = sum(recall_at_k.values()) / len(recall_at_k)
#     avg_map_at_k = sum(map_at_k.values()) / len(map_at_k)
        
#     return avg_precision_at_k, avg_recall_at_k, avg_map_at_k

###############################################################################################

# def calculate_kfold_score(recommendation_sys, cosine_similarity_matrix, train_data, valid_data, all_user_toread, all_books, n_sample, top_k):

#     """
#     Function to calculate precision, recall and mean average precision given training/validation data.

#     Parameters:
#     - recommendation_sys: Chosen recommendation system (e.g. popularity_rec)
#     - cosine_similarity_matrix: Each row corresponds to a user and each column corresponds to a book
#     - train_data: List of user ratings for each book
#     - valid_data: List of user ratings for each book
#     - all_user_toread: DataFrame containing to read list for all users.
#     - all_books: DataFrame containing metadata for all books.
#     - n_sample: Number of recommendations to generate.
#     - top_k: Number of top recommendations to consider
    
#     Returns:
#     - overall_precision: Average precision@K across users in the validation data
#     - overall_recall: Average Recall@k across users in the validation data
#     - overall_map: Average Mean average precision @ k across users in the validation data
#     """
#     user_recommendations = {}
#     user_history = {}
#     user_precision = {}
#     user_recall = {}
#     user_map = {}

#     # Randomly sample user ids from the validation set
#     user_ids = list(valid_data["user_id"].unique())
#     sampled_user_ids = random.sample(user_ids, int(0.001*len(user_ids)))

#     for user_id in sampled_user_ids:
        
#         # Output a dictionary containing n_sample predicted items for each user.
#         user_recommendations[user_id] = recommendation_sys(user_id, cosine_similarity_matrix, train_data, all_user_toread, all_books, n_sample)
    
#         # Output a dictionary containing actual rated items for each user.
#         user_history[user_id] = valid_data.loc[valid_data["user_id"] == id, "book_id"].tolist()

#         # Get lists of predicted & actual items for the user, default to an empty list
#         predicted = user_recommendations.get(user_id, [])
#         actual = user_history.get(user_id, [])
    
#         # Output a dictionary containing metric scores for each user 
#         user_precision[id] = calculate_precision_at_k(actual, predicted, top_k)
#         user_recall[id] = calculate_recall_at_k(actual, predicted, top_k)
#         user_map[id] = calculate_mean_average_precision_at_k(actual, predicted, top_k)
        
#     # Average metric over all users
#     overall_precision = sum(user_precision.values()) / len(user_precision)
#     overall_recall = sum(user_recall.values()) / len(user_recall)
#     overall_map = sum(user_map.values()) / len(user_map)

#     return overall_precision, overall_recall, overall_map

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