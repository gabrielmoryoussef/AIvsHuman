import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from topicSCORE import topic_score
import re
import os
from scipy.optimize import minimize
from typing import Union, Dict, Tuple
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse
import traceback
from Classifiers.PHC import get_pvals as phc_get_pvals, accuracy as phc_accuracy, clean_text as phc_clean_text, get_vocab
from topicSCORE import print_top_words_per_topic
from utils import *
from openai import OpenAI
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def create_vocabulary(df: pd.DataFrame, human_col: str = 'human', 
                     ai_col: str = 'ChatGPT'):
    # Combine all text to create common vocabulary
    all_text = pd.concat([df[human_col], df[ai_col]]).apply(preprocess_text)
    
    # Create and fit vectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(all_text)
    
    return vectorizer

def create_document_word_matrices(df, vectorizer, 
                                human_col = 'human', 
                                ai_col = 'ChatGPT'):
    # Preprocess texts
    human_texts = df[human_col].apply(preprocess_text)
    ai_texts = df[ai_col].apply(preprocess_text)
    
    # Transform texts to document-word matrices
    D_h = vectorizer.transform(human_texts)
    D_ai = vectorizer.transform(ai_texts)
    
    # Convert to frequency matrices (normalize by document length)
    D_h_freq = D_h.copy()
    D_ai_freq = D_ai.copy()
    
    # Normalize each document (column) by its length
    doc_lengths_h = np.array(D_h.sum(axis=0))[0]
    doc_lengths_ai = np.array(D_ai.sum(axis=0))[0]
    
    for i in range(D_h.shape[1]):
        if doc_lengths_h[i] > 0:
            D_h_freq[:, i] = D_h_freq[:, i] / doc_lengths_h[i]
            
    for i in range(D_ai.shape[1]):
        if doc_lengths_ai[i] > 0:
            D_ai_freq[:, i] = D_ai_freq[:, i] / doc_lengths_ai[i]
    
    return D_h_freq, D_ai_freq


def filter_words(A_h, A_ai, t, vocabulary = None):
    # Compute L1 norm of differences for each word
    differences = np.linalg.norm(A_h - A_ai, ord=1, axis=1)
    
    # Find indices where difference exceeds threshold
    selected_indices = np.where(differences > t)[0]
    
    # Create filtered matrices
    A_hS = A_h[selected_indices, :]
    A_aiS = A_ai[selected_indices, :]
    
    result = {
        'A_hS': A_hS,
        'A_aiS': A_aiS,
        'selected_indices': selected_indices,
        'differences': differences[selected_indices]
    }
    
    # If vocabulary is provided, include selected words
    if vocabulary is not None:
        result['selected_vocabulary'] = vocabulary[selected_indices]
        
        # Create a sorted version by difference magnitude
        sorted_indices = np.argsort(-differences[selected_indices])
        result['sorted_words'] = [
            (vocabulary[selected_indices[i]], 
             differences[selected_indices[i]]) 
            for i in sorted_indices
        ]
    
    return result

def solve_reconstruction(A, d):
    K = A.shape[1]
    
    # Objective function: ||Aw - d||_2^2
    def objective(w):
        return np.linalg.norm(A @ w - d)**2
    
    # Constraint: ||w||_1 = 1
    def l1_constraint(w):
        return np.sum(np.abs(w)) - 1
    
    # Try different initial points
    best_result = None
    best_obj = float('inf')
    
    # List of initial points to try
    initial_points = [
        np.ones(K) / K,  # Uniform weights
        np.random.dirichlet(np.ones(K)),  # Random point on simplex
        np.zeros(K),  # Zeros with one one
    ]
    initial_points[-1][0] = 1.0
    constraint = {
        'type': 'eq',        # Equality constraint
        'fun': l1_constraint # Constraint function
    }
    for w0 in initial_points:
        try:
            result = minimize(
                objective,
                w0,
                method='SLSQP',
                constraints=[constraint],
                bounds=[(0, None) for _ in range(K)],
                options={
                    'ftol': 1e-6,        # Reduced precision requirement
                    'maxiter': 5000,      # Increased max iterations
                    'disp': False,
                    'eps': 1e-8           # Step size for finite difference
                }
            )
            
            # If this is the best result so far, keep it
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
                
        except Exception as e:
            continue
    
    # If all optimizations failed, return basic solution
    if best_result is None:
        w = np.ones(K) / K
        return objective(w), w
    
    # Ensure weights sum to 1 and are non-negative
    w = np.maximum(best_result.x, 0)
    w = w / (np.sum(w) + 1e-10)
    
    return objective(w), w

def eval_document(d, 
                 A_hS, 
                 A_aiS, 
                 selected_indices,
                 return_details = False,
                 debug = False):
    # Filter document to keep only selected words
    d_S = d[selected_indices]
    
    # Normalize document vector (if not already normalized)
    if np.sum(d_S) != 0:
        d_S = d_S / np.sum(d_S)
    
    # Compute reconstruction errors for both models
    error_human, weights_human = solve_reconstruction(A_hS, d_S)
    error_ai, weights_ai = solve_reconstruction(A_aiS, d_S)
    
    if debug:
        print(f"Human error: {error_human:.6f}")
        print(f"AI error: {error_ai:.6f}")
        print(f"Difference (human - ai): {error_human - error_ai:.6f}")
        print(f"Human weights sum: {np.sum(weights_human):.6f}")
        print(f"AI weights sum: {np.sum(weights_ai):.6f}")
    
    # Classify based on minimum reconstruction error
    classification = 'human' if error_human < error_ai else 'ChatGPT'
    
    if return_details:
        return {
            'classification': classification,
            'error_human': error_human,
            'error_ai': error_ai,
            'weights_human': weights_human,
            'weights_ai': weights_ai,
            'd_S': d_S
        }
    
    return classification

def prepare_and_eval_text(text, 
                         vectorizer,
                         A_hS, 
                         A_aiS,
                         selected_indices,
                         return_details = False):
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Transform to document vector
    d = vectorizer.transform([processed_text]).toarray()[0]
    
    # Evaluate
    return eval_document(d, A_hS, A_aiS, selected_indices, return_details)



def get_author_pairs(df, N, min_samples=15):
    """
    Select N authors with minimum samples and return all possible pairs
    """
    # Count samples per author
    author_counts = df['author'].value_counts()
    eligible_authors = author_counts[author_counts >= min_samples].index.tolist()
    
    if len(eligible_authors) < N:
        raise ValueError(f"Need at least {N} different authors with {min_samples} samples. Found {len(eligible_authors)} eligible authors.")
    
    # Randomly select N authors
    selected_authors = np.random.choice(eligible_authors, size=N, replace=False)
    
    # Generate all possible pairs
    author_pairs = list(itertools.combinations(selected_authors, 2))
    
    return author_pairs

           
def get_eligible_authors(df, min_samples=30):
    """
    Select authors with minimum number of samples
    """
    author_counts = df['author'].value_counts()
    return author_counts[author_counts >= min_samples].index.tolist()

if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Text Classification with Topic SCORE')
    parser.add_argument('--no-cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--K', type=int, default=3, help='Number of topics (if no CV)')
    parser.add_argument('--threshold', type=float, default=0.0001, help='Threshold value (if no CV)')
    parser.add_argument('--compare-phc', action='store_true', help='Compare with PHC method')
    parser.add_argument('--same-weight', action='store_true', help='Use same weight for human and AI matrices')
    parser.add_argument('--num-authors', type=int, default=10, help='Number of authors to select')
    parser.add_argument('--compare-rf', action='store_true', help='Compare with Random Forest method')
    parser.add_argument('--tf', action='store_true', help='Compare with TF-IDF method')
    parser.add_argument('--st-model', type=str, default='all-MiniLM-L6-v2', help='Sentence Transformer model name')
    parser.add_argument('--llm', action='store_true', help='Include LLM (GPT-4-mini) classification')
    parser.add_argument('--ts', action='store_true', help='Include topic score')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--output', type=str, default='deepseek_C2_r1.csv', help='OpenAI model to use')
    parser.add_argument('--compare-ts-llm', action='store_true', 
                   help='Compare with TopicScore + LLM method')
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('output_no_french.csv')
    df2 = pd.read_csv("single_GPT.csv")

    
    # # Get eligible authors and select N random ones
    # eligible_authors = get_eligible_authors(df, min_samples=30)
    # if len(eligible_authors) < args.num_authors:
    #     raise ValueError(f"Need at least {args.num_authors} authors with 30+ samples. Found {len(eligible_authors)}.")
    
    # selected_authors = np.random.choice(eligible_authors, size=args.num_authors, replace=False)
    # print(f"Selected {len(selected_authors)} authors for evaluation")
    selected_authors = list(df2["author"])

    
    # Create DataFrame to store results
    results_data = []
    all_results = []
    print(f"Starting the clock for {args.model}")
    t1 = time.time()
    
    # Process each author
    for author in selected_authors:
        for rs in range(1,20):
        print("-"*100)
        print(f"Processing author: {author}")
        
        try:
            # # Get author's data and create train/test split
            author_data = df[df['author'] == author].copy()
            train_indices, test_indices = train_test_split(
                range(len(author_data)), 
                test_size=0.2, 
                random_state=rs
            )
            
            df_train = author_data.iloc[train_indices]
            df_test = author_data.iloc[test_indices]
            test_true_labels = ['human'] * len(df_test) + ['ChatGPT'] * len(df_test)
            # Get author's data
            # if author=="Stephane Girard":
            #     author_data = df[df["author"] == 'Stéphane Girard'].copy()
            # else:
            #     author_data = df[df["author"] == author].copy()
            # Use all data as test set
            df_test = author_data
            

            # Create labels array - assuming you want to label half as human and half as ChatGPT
            half_length = len(df_test)
            # test_true_labels = ['human'] * half_length + ['ChatGPT'] * half_length
            # #print(f"Train size: {len(df_train)}")
            # print(f"Test size: {len(df_test)}")
            
            if args.ts:
                # Create stop words list
                stop_words = list(ENGLISH_STOP_WORDS)
                additional_stops = ['Abstract']
                stop_words.extend(additional_stops)

                # Create and fit shared vectorizer
                print("\nCreating shared vocabulary...")
                
                # Fit vectorizer on training data only
                train_texts = df_train['human'].tolist() + df_train['ChatGPT'].tolist()
                shared_vectorizer = CountVectorizer(
                    stop_words=stop_words,
                    min_df=5,
                    max_df=0.95,
                    token_pattern=r'(?u)\b\w+\b'
                )
                train_texts = [' '.join([word for word in text.split() 
                                    if len(word) >= 5 and not word.isdigit()]) 
                            for text in train_texts]
                shared_vectorizer.fit(train_texts)
                vocabulary = shared_vectorizer.get_feature_names_out()
                print(f"Vocabulary size: {len(vocabulary)}")

                # Parameter selection
                if args.no_cv:
                    print(f"\nSkipping CV, using specified parameters: K={args.K}, threshold={args.threshold}")
                    best_K = args.K
                    best_threshold = args.threshold
                else:
                    print("\nPerforming cross-validation for parameter selection...")
                    
                    # Create train/val split from training data
                    kf = KFold(n_splits=5, shuffle=True, random_state=10)
                    
                    # Define parameter grid
                    topic_numbers = [3, 4, 5, 6]
                    thresholds = [0.1, 0.01, 0.005, 0.0001, 0.00004]
                    
                    cv_results = []
                    best_score = 0
                    best_K = None
                    best_threshold = None
                    
                    for K in topic_numbers:
                        for threshold in thresholds:
                            print(f"\nTesting K={K}, threshold={threshold}")
                            fold_scores = []
                            
                            for fold, (train_idx, val_idx) in enumerate(kf.split(df_train), 1):
                                try:
                                    # Split training data into train/val for this fold
                                    df_fold_train = df_train.iloc[train_idx]
                                    df_fold_val = df_train.iloc[val_idx]
                                    
                                    # Process fold training texts
                                    fold_train_texts = (df_fold_train['human'].tolist() + 
                                                    df_fold_train['ChatGPT'].tolist())
                                    
                                    # Transform and normalize training data
                                    X_fold_train = shared_vectorizer.transform(fold_train_texts).toarray()
                                    doc_lengths = X_fold_train.sum(axis=1)
                                    X_fold_train = X_fold_train / doc_lengths[:, np.newaxis]
                                    X_fold_train = X_fold_train.T
                                    
                                    # Split into human and ChatGPT matrices
                                    n_fold_train = len(df_fold_train)
                                    D_h = X_fold_train[:,:n_fold_train]
                                    D_ai = X_fold_train[:,n_fold_train:]
                                    
                                    if args.same_weight:
                                        # Stack matrices vertically and normalize by 2
                                        D_combined = np.vstack([D_h, D_ai]) / 2
                                        results_combined = topic_score(K, D_combined)
                                        p = D_h.shape[0]
                                        A_combined = results_combined['A_hat']
                                        A_h = A_combined[:p, :]
                                        A_ai = A_combined[p:, :]
                                    else:
                                        results_h = topic_score(K, D_h)
                                        results_ai = topic_score(K, D_ai)
                                        A_h = results_h['A_hat']
                                        A_ai = results_ai['A_hat']
                                        
                                    # Filter words
                                    filtered_results = filter_words(A_h, A_ai, threshold, vocabulary)
                                    
                                    if len(filtered_results['selected_indices']) == 0:
                                        print(f"Fold {fold}: No words selected, skipping...")
                                        continue
                                    
                                    A_hS = filtered_results['A_hS']
                                    A_aiS = filtered_results['A_aiS']
                                    selected_indices = filtered_results['selected_indices']
                                    
                                    # Validation predictions
                                    val_preds = []
                                    for source in ['human', 'ChatGPT']:
                                        val_matrix = shared_vectorizer.transform(
                                            df_fold_val[source]).toarray()
                                        val_lengths = val_matrix.sum(axis=1)
                                        val_matrix = val_matrix / val_lengths[:, np.newaxis]
                                        
                                        for doc in val_matrix:
                                            result = eval_document(doc, A_hS, A_aiS, selected_indices)
                                            val_preds.append(result)
                                    
                                    val_true = (['human'] * len(df_fold_val) + 
                                            ['ChatGPT'] * len(df_fold_val))
                                    fold_acc = accuracy_score(val_true, val_preds)
                                    fold_scores.append(fold_acc)
                                    print(f"Fold {fold} accuracy: {fold_acc:.4f}")
                                    
                                except Exception as e:
                                    print(f"Error in fold {fold}: {str(e)}")
                                    continue
                            
                            if fold_scores:
                                avg_score = np.mean(fold_scores)
                                std_score = np.std(fold_scores)
                                cv_results.append({
                                    'K': K,
                                    'threshold': threshold,
                                    'mean_accuracy': avg_score,
                                    'std_accuracy': std_score,
                                    'n_folds': len(fold_scores)
                                })
                                print(f"Average accuracy: {avg_score:.4f} (±{std_score:.4f})")
                                
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_K = K
                                    best_threshold = threshold
                    
                    if cv_results:
                        cv_df = pd.DataFrame(cv_results)
                        print("\nCross-validation results:")
                        print(cv_df)
                        print(f"\nBest parameters: K={best_K}, threshold={best_threshold}")
                        print(f"Best CV accuracy: {best_score:.4f}")
                        # Save CV results with author info
                        cv_df['author'] = author
                        cv_df.to_csv(f'cv_results_{author}.csv', index=False)
                    else:
                        print("No valid CV results, using default parameters")
                        best_K = 15
                        best_threshold = 0.01

                # Train final model using best parameters
                print(f"\nTraining final model with K={best_K}, threshold={best_threshold}...")
                
                # Transform and normalize all training data
                X_train = shared_vectorizer.transform(train_texts).toarray()
                doc_lengths = X_train.sum(axis=1)
                X_train = X_train / doc_lengths[:, np.newaxis]
                X_train = X_train.T
                
                # Split into human and ChatGPT matrices
                n_train = len(df_train)
                D_h = X_train[:,:n_train]
                D_ai = X_train[:,n_train:]

                if args.same_weight:
                    print("Stacking matrices")
                    D_combined = np.vstack([D_h, D_ai]) / 2
                    results_combined = topic_score(int(best_K), D_combined)
                    p = D_h.shape[0]
                    A_combined = results_combined['A_hat']
                    A_h = A_combined[:p, :]
                    A_ai = A_combined[p:, :]
                else:
                    results_h = topic_score(int(best_K), D_h)
                    results_ai = topic_score(int(best_K), D_ai)
                    A_h = results_h['A_hat']
                    A_ai = results_ai['A_hat']
                
                # Filter words using best threshold
                filtered_results = filter_words(A_h, A_ai, best_threshold, vocabulary)
                topic_score_words = set([w for w, _ in filtered_results['sorted_words']])
                print(f"\nTopic SCORE unique words used: {len(topic_score_words)}")
                
                A_hS = filtered_results['A_hS']
                A_aiS = filtered_results['A_aiS']
                selected_indices = filtered_results['selected_indices']
                
                # Evaluate on test set
                print("\nEvaluating on test set...")
                test_predictions = []
                test_details = []
                
                # Process test documents
                for source in ['human', 'ChatGPT']:
                    test_matrix = shared_vectorizer.transform(df_test[source]).toarray()
                    test_lengths = test_matrix.sum(axis=1)
                    test_matrix = test_matrix / test_lengths[:, np.newaxis]
                    
                    for doc in test_matrix:
                        result = eval_document(doc, A_hS, A_aiS, selected_indices, 
                                            return_details=True)
                        test_predictions.append(result['classification'])
                        test_details.append(result)
                
                # Calculate TopicScore metrics
                topic_score_accuracy = accuracy_score(test_true_labels, test_predictions)
                topic_score_cm = confusion_matrix(test_true_labels, test_predictions)
                
                methods_results = {
                    'Topic SCORE': {
                        'accuracy': topic_score_accuracy,
                        'confusion_matrix': topic_score_cm,
                        'predictions': test_predictions
                    }
                }

            # Compare with PHC if requested
            if args.compare_phc:
                print("\nComparing with PHC method...")
                
                # Prepare data for PHC
                phc_train = pd.DataFrame({
                    'author': ['author1']*len(df_train) + ['author2']*len(df_train),
                    'title': [''] * (2*len(df_train)),
                    'text': list(df_train['human']) + list(df_train['ChatGPT'])
                })
                
                phc_test = pd.DataFrame({
                    'author': ['author1']*len(df_test) + ['author2']*len(df_test),
                    'title': [''] * (2*len(df_test)),
                    'text': list(df_test['human']) + list(df_test['ChatGPT'])
                })
                
                # Clean data and get PHC predictions
                print("Cleaning data for PHC...")
                phc_train_clean = phc_clean_text(phc_train)
                phc_test_clean = phc_clean_text(phc_test)
                
                # Run PHC with shared vocabulary
                print("Running PHC classification...")
                phc_preds, phc_true, selected_words_info = phc_get_pvals(
                    phc_train_clean, 
                    phc_test_clean, 
                    show_hist=False,
                    vocab=vocabulary
                )
                
                # Debug PHC output
                print("PHC predictions sample:", phc_preds[:5])
                print("PHC true labels sample:", phc_true[:5])
                print("Unique PHC predictions:", set(phc_preds))
                print("Unique PHC true labels:", set(phc_true))

                # Convert binary predictions to string labels
                try:
                    # Map 0 to 'human' and 1 to 'ChatGPT'
                    phc_preds_str = ['ChatGPT' if pred == 1 else 'human' for pred in phc_preds]
                    phc_true_str = ['ChatGPT' if label == 1 else 'human' for label in phc_true]
                    
                    phc_accuracy_ = phc_accuracy(phc_preds, phc_true)
                    phc_cm = confusion_matrix(phc_true_str, phc_preds_str, labels=['human', 'ChatGPT'])
                    
                    methods_results['PHC'] = {
                        'accuracy': phc_accuracy_,
                        'predictions': phc_preds_str,
                        'confusion_matrix': phc_cm
                    }
                except Exception as e:
                    print(f"Error calculating PHC metrics: {str(e)}")
                    print("Skipping PHC method for this author")

            # Random Forest comparison if requested
            if args.compare_rf:
                print("\nComparing with Sentence Transformer + Random Forest method...")
                
                try:
                    st_rf_results = sentence_transformer_random_forest(
                        df_train, 
                        df_test,
                        model_name=args.st_model,
                        n_estimators=100,
                        batch_size=32
                    )
                    
                    methods_results['Random Forest + ST'] = {
                        'accuracy': st_rf_results['accuracy'],
                        'confusion_matrix': st_rf_results['confusion_matrix'],
                        'predictions': st_rf_results['predictions']
                    }
                    
                except Exception as e:
                    print(f"Error in Random Forest comparison: {str(e)}")
                    traceback.print_exc()

                # Store results for this author
                author_results = {
                    'author': author,
                    'train_size': len(df_train),
                    'test_size': len(df_test),
                    'best_K': best_K,
                    'best_threshold': best_threshold
                }
            
            # # Add results for each method
            # for method, results in methods_results.items():
            #     author_results[f'{method}_accuracy'] = results['accuracy']
            #     if 'confusion_matrix' in results:
            #         cm = results['confusion_matrix']
            #         author_results[f'{method}_true_positive'] = cm[0][0]
            #         author_results[f'{method}_false_positive'] = cm[0][1]
            #         author_results[f'{method}_false_negative'] = cm[1][0]
            #         author_results[f'{method}_true_negative'] = cm[1][1]
            
            # Compare with TopicScore + LLM if requested


            # Compare with TF-IDF if requested
            if args.tf:
                print("\nComparing with TF-IDF classifier...")
                try:
                    # Create and fit TF-IDF vectorizer on training data

                    tfidf = TfidfVectorizer(
                        stop_words='english',
                        min_df=5,
                        max_df=0.95,
                        token_pattern=r'(?u)\b\w+\b'
                    )

                    # Prepare training data
                    X_train_tfidf = tfidf.fit_transform(
                        list(df_train['human']) + list(df_train['ChatGPT'])
                    )
                    y_train_tfidf = ['human'] * len(df_train) + ['ChatGPT'] * len(df_train)

                    # Train logistic regression classifier
                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    clf.fit(X_train_tfidf, y_train_tfidf)

                    # Prepare test data
                    X_test_tfidf = tfidf.transform(
                        list(df_test['human']) + list(df_test['ChatGPT'])
                    )
                    
                    # Make predictions
                    tfidf_predictions = clf.predict(X_test_tfidf)

                    # Calculate metrics
                    tfidf_accuracy = accuracy_score(test_true_labels, tfidf_predictions)
                    tfidf_cm = confusion_matrix(test_true_labels, tfidf_predictions)

                    # Add results to methods_results dictionary
                    methods_results['TF-IDF'] = {
                        'accuracy': tfidf_accuracy,
                        'confusion_matrix': tfidf_cm,
                        'predictions': tfidf_predictions
                    }

                    # Store results in author_results
                    author_results['TF-IDF_accuracy'] = tfidf_accuracy
                    author_results['TF-IDF_true_positive'] = tfidf_cm[0][0]
                    author_results['TF-IDF_false_positive'] = tfidf_cm[0][1]
                    author_results['TF-IDF_false_negative'] = tfidf_cm[1][0]
                    author_results['TF-IDF_true_negative'] = tfidf_cm[1][1]

                    print(f"TF-IDF Classification Accuracy: {tfidf_accuracy:.4f}")

                except Exception as e:
                    print(f"Error in TF-IDF classification: {str(e)}")
                    traceback.print_exc()
            if args.compare_ts_llm:
                print(f"\nRunning TopicScore + {args.model} classifier...")
                test_docs = list(df_test['human']) + list(df_test['ChatGPT'])
                
                try:
                    # Get discriminative words from the already computed Topic Score results
                    sorted_words = filtered_results['sorted_words']  # List of (word, score) tuples
                    
                    # Split words into human and ChatGPT based on their topic distributions
                    h_sums = np.sum(A_h[selected_indices, :], axis=1)
                    ai_sums = np.sum(A_ai[selected_indices, :], axis=1)
                    
                    human_words = [(vocabulary[selected_indices[i]], filtered_results['differences'][i])
                                for i in range(len(selected_indices)) 
                                if h_sums[i] > ai_sums[i]]
                    chatgpt_words = [(vocabulary[selected_indices[i]], filtered_results['differences'][i])
                                    for i in range(len(selected_indices)) 
                                    if h_sums[i] <= ai_sums[i]]
                    
                    # Sort words by their importance (difference score)
                    human_words = sorted(human_words, key=lambda x: x[1], reverse=True)[:10]
                    chatgpt_words = sorted(chatgpt_words, key=lambda x: x[1], reverse=True)[:10]
                    
                    # Create prompt with Topic Score insights
                    human_word_list = ', '.join([w[0] for w in human_words])
                    ai_word_list = ', '.join([w[0] for w in chatgpt_words])
                    
                    discriminative_info = (
                        f"\n\nBased on topic analysis, here are characteristic words:"
                        f"\nTypically human-written: {human_word_list}"
                        f"\nTypically AI-written: {ai_word_list}"
                    )
                    #print("system prompt = ", discriminative_info)
                    system_prompt = f"""You are a classifier that determines whether text is human-written or written by a human but AI-edited. 
    You will be given academic abstracts that are either written by humans or edited by AI.{discriminative_info}
    
    Respond with exactly one word: either 'human' for human-written text or 'ChatGPT' for AI-edited text. 
    Be as accurate as possible in your classification, considering both the writing style and the presence of characteristic words."""
                    print("TS prompt = ", system_prompt)
                    # Make predictions using OpenAI API
                    client = OpenAI()
                    ts_llm_predictions = []
                    
                    for i, text in enumerate(test_docs):
                        if i % 10 == 0:
                            print(f"Processing document {i+1}/{len(test_docs)}")

                            
                        try:
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Classify this text: {text}"}
                            ]
                            
                            response = client.chat.completions.create(
                                model=args.model,
                                messages=messages,
                                max_tokens=1,
                                temperature=0,
                                stream=False
                            )
                            
                            pred = response.choices[0].message.content.strip().lower()
                            if pred not in ['human', 'chatgpt']:
                                print(f"Warning: Unexpected prediction '{pred}', defaulting to 'ChatGPT'")
                                pred = 'ChatGPT'
                            
                            ts_llm_predictions.append(pred)
                            
                        except Exception as e:
                            # Fallback to Topic Score prediction for this document
                            print(f"LLM failed: {str(e)}. Using Topic Score fallback.")
                            d = shared_vectorizer.transform([text]).toarray()[0]
                            d = d / (np.sum(d) + 1e-10)
                            result = eval_document(d, A_hS, A_aiS, selected_indices)
                            ts_llm_predictions.append(result)
                    
                    # Calculate metrics
                    ts_llm_accuracy = accuracy_score(test_true_labels, ts_llm_predictions)
                    ts_llm_cm = confusion_matrix(test_true_labels, ts_llm_predictions)
                    
                    # Add results to methods_results dictionary
                    methods_results['TopicScore_LLM'] = {
                        'accuracy': ts_llm_accuracy,
                        'confusion_matrix': ts_llm_cm,
                        'predictions': ts_llm_predictions
                    }
                    
                    # Store results in author_results
                    author_results['TopicScore_LLM_accuracy'] = ts_llm_accuracy
                    author_results['TopicScore_LLM_true_positive'] = ts_llm_cm[0][0]
                    author_results['TopicScore_LLM_false_positive'] = ts_llm_cm[0][1]
                    author_results['TopicScore_LLM_false_negative'] = ts_llm_cm[1][0]
                    author_results['TopicScore_LLM_true_negative'] = ts_llm_cm[1][1]
                
                except Exception as e:
                    print(f"Error in TopicScore + LLM classification: {str(e)}")
                    traceback.print_exc()
            
            # Add LLM results if requested
            if args.llm:
                print(f"\nRunning {args.model} classifiers...")
                test_docs = list(df_test['human']) + list(df_test['ChatGPT'])
                
                try:
                    # Run LLM with PHC words if PHC was run
                    if args.compare_phc:
                        print("Running LLM with PHC discriminative words...")
                        llm_phc_results = llm_classifier(
                            test_docs,
                            selected_words_info=selected_words_info,
                            model=args.model
                        )
                        
                        llm_phc_accuracy = accuracy_score(test_true_labels, llm_phc_results['predictions'])
                        llm_phc_cm = confusion_matrix(test_true_labels, llm_phc_results['predictions'])
                    
                        
                        # Store PHC-guided LLM results
                        author_results['LLM_PHC_accuracy'] = llm_phc_accuracy
                        author_results['LLM_PHC_true_positive'] = llm_phc_cm[0][0]
                        author_results['LLM_PHC_false_positive'] = llm_phc_cm[0][1]
                        author_results['LLM_PHC_false_negative'] = llm_phc_cm[1][0]
                        author_results['LLM_PHC_true_negative'] = llm_phc_cm[1][1]
                    
                    # Run pure LLM (without PHC words)
                    print("Running pure LLM classifier...")
                    file_path = f"{author}_c2_corrected.csv"

                    # Check if file exists
                    if os.path.exists(file_path):
                        # Load existing CSV
                        
                        df_aut = pd.read_csv(file_path)
                        print("Existing author.csv found. Adding model predictions.")
                    else:
                        # Create new DataFrame if file doesn't exist
                        df_aut = pd.DataFrame()
                        print("Creating new author.csv with model predictions.")
                    llm_pure_results = llm_classifier(
                        test_docs,
                        selected_words_info=None,  # No PHC words
                        model=args.model
                    )
                    if 'reasoner' in args.model or "nvidia" in args.model:
                        llm_pure_accuracy = accuracy_score(test_true_labels, llm_pure_results['predictions'])
                        print("acc = ",llm_pure_accuracy)
                        llm_pure_cm = confusion_matrix(test_true_labels, llm_pure_results['predictions'])
                        print("sooooo far so gooood")
                        thoughts = llm_pure_results['thoughts']
                        df_ = pd.DataFrame({'R1 thoughts': thoughts})
                        df_.to_csv(f"{author}_thought_c1_corrected.csv")
                        # Create a DataFrame with a single row index
                        llm_res = {"author":author,'LLM_Pure_accuracy': llm_pure_accuracy}
                        all_results.append(llm_res)
                        df_aut[f"{args.model}"] = llm_pure_results['predictions']
                        df_aut.to_csv(file_path, index=False)
                        
                    else:
                        df_aut[f"{args.model}"] = llm_pure_results['predictions']
                        df_aut.to_csv(file_path, index=False)
                        llm_pure_accuracy = accuracy_score(test_true_labels, llm_pure_results['predictions'])
                        print("acc = ", llm_pure_accuracy)
                        llm_pure_cm = confusion_matrix(test_true_labels, llm_pure_results['predictions'])
                        llm_res = {"author":author,'LLM_Pure_accuracy': llm_pure_accuracy}
                        all_results.append(llm_res)
                        
                        
                    # Store pure LLM results
                    #llm_res = {}
                    # author_results['LLM_Pure_accuracy'] = llm_pure_accuracy
                    # author_results['LLM_Pure_true_positive'] = llm_pure_cm[0][0]
                    # author_results['LLM_Pure_false_positive'] = llm_pure_cm[0][1]
                    # author_results['LLM_Pure_false_negative'] = llm_pure_cm[1][0]
                    # author_results['LLM_Pure_true_negative'] = llm_pure_cm[1][1]
                    #llm_res['LLM_Pure_accuracy'] = llm_pure_accuracy
                    # llm_res['LLM_Pure_true_positive'] = llm_pure_cm[0][0]
                    # llm_res['LLM_Pure_false_positive'] = llm_pure_cm[0][1]
                    # llm_res['LLM_Pure_false_negative'] = llm_pure_cm[1][0]
                    # llm_res['LLM_Pure_true_negative'] = llm_pure_cm[1][1]

                    
                except Exception as e:
                    print(f"Error in LLM classification: {str(e)}")
                    traceback.print_exc()
            
            
            # # Append results for this author
            # results_data.append(author_results)
            
            # # Save intermediate results after each author
            # results_df = pd.DataFrame(results_data)
            # results_df.to_csv(args.output, index=False)
            # print(f"\nIntermediate results saved to {args.output}")
            # print(time.sleep(30))
            
        except Exception as e:
            print(f"Error processing author {author}: {str(e)}")
            traceback.print_exc()
            continue
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output, index=False)
    t2 = time.time()
    print(f"Elapsed time for {args.model}: {t2 - t1}")
    
    # # Final summary
    # if results_data:
    #     final_df = pd.DataFrame(results_data)
    #     print("\nFinal Results Summary:")
    #     print("="*50)
        
    #     # Calculate average accuracy for each method
    #     methods = [col.replace('_accuracy', '') 
    #               for col in final_df.columns 
    #               if col.endswith('_accuracy')]
                  
    #     print("\nAverage Accuracy by Method:")
    #     for method in methods:
    #         avg_acc = final_df[f'{method}_accuracy'].mean()
    #         std_acc = final_df[f'{method}_accuracy'].std()
    #         print(f"{method}: {avg_acc:.4f} (±{std_acc:.4f})")
            
    #     # Save detailed results
    #     final_df.to_csv('final_author_results_single.csv', index=False)
    #     print("\nFinal results saved to 'final_author_results_single.csv'")
        
    #     # Create summary statistics
    #     summary_stats = {
    #         'total_authors': len(final_df),
    #         'avg_train_size': final_df['train_size'].mean(),
    #         'avg_test_size': final_df['test_size'].mean(),
    #         'min_train_size': final_df['train_size'].min(),
    #         'max_train_size': final_df['train_size'].max()
    #     }
        
    #     # Add method-specific statistics
    #     for method in methods:
    #         summary_stats[f'{method}_avg_accuracy'] = final_df[f'{method}_accuracy'].mean()
    #         summary_stats[f'{method}_std_accuracy'] = final_df[f'{method}_accuracy'].std()
    #         summary_stats[f'{method}_min_accuracy'] = final_df[f'{method}_accuracy'].min()
    #         summary_stats[f'{method}_max_accuracy'] = final_df[f'{method}_accuracy'].max()
            
    #         if f'{method}_true_positive' in final_df.columns:
    #             tp = final_df[f'{method}_true_positive'].sum()
    #             fp = final_df[f'{method}_false_positive'].sum()
    #             fn = final_df[f'{method}_false_negative'].sum()
    #             tn = final_df[f'{method}_true_negative'].sum()
                
    #             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    #             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    #             f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
    #             summary_stats[f'{method}_overall_precision'] = precision
    #             summary_stats[f'{method}_overall_recall'] = recall
    #             summary_stats[f'{method}_overall_f1'] = f1
        
    #     # Save summary statistics
    #     pd.DataFrame([summary_stats]).to_csv(args.output, index=False)
    #     print(f"\nSummary statistics saved to {args.output}")