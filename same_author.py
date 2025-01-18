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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse
import traceback
from Classifiers.PHC import get_pvals as phc_get_pvals, accuracy as phc_accuracy, clean_text as phc_clean_text, get_vocab
from topicSCORE import print_top_words_per_topic
from utils import *
from openai import OpenAI
import time
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
    parser.add_argument('--compare-phc', default=True, help='Compare with PHC method')
    parser.add_argument('--same-weight', action='store_true', help='Use same weight for human and AI matrices')
    parser.add_argument('--num-authors', type=int, default=15, help='Number of authors to select')
    parser.add_argument('--compare-rf', action='store_true', help='Compare with Random Forest method')
    parser.add_argument('--st-model', type=str, default='all-MiniLM-L6-v2', help='Sentence Transformer model name')
    parser.add_argument('--llm', action='store_true', help='Include LLM (GPT-4-mini) classification')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--output', type=str, default='author_results.csv', help='OpenAI model to use')
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('new_data_gpt_only.csv')
    
    # Get eligible authors and select N random ones
    eligible_authors = get_eligible_authors(df, min_samples=30)
    if len(eligible_authors) < args.num_authors:
        raise ValueError(f"Need at least {args.num_authors} authors with 30+ samples. Found {len(eligible_authors)}.")
    
    selected_authors = np.random.choice(eligible_authors, size=args.num_authors, replace=False)
    print(f"Selected {len(selected_authors)} authors for evaluation")
    
    # Create DataFrame to store results
    results_data = []
    
    # Process each author
    for author in selected_authors:
        print("-"*100)
        print(f"Processing author: {author}")
        
        try:
            # Get author's data and create train/test split
            author_data = df[df['author'] == author].copy()
            train_indices, test_indices = train_test_split(
                range(len(author_data)), 
                test_size=0.2, 
                random_state=42
            )
            
            df_train = author_data.iloc[train_indices]
            df_test = author_data.iloc[test_indices]
            
            print(f"Train size: {len(df_train)}")
            print(f"Test size: {len(df_test)}")
            
            # Create stop words list
            stop_words = list(text.ENGLISH_STOP_WORDS)
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
            test_true_labels = ['human'] * len(df_test) + ['ChatGPT'] * len(df_test)
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
            
            # Add results for each method
            for method, results in methods_results.items():
                author_results[f'{method}_accuracy'] = results['accuracy']
                if 'confusion_matrix' in results:
                    cm = results['confusion_matrix']
                    author_results[f'{method}_true_positive'] = cm[0][0]
                    author_results[f'{method}_false_positive'] = cm[0][1]
                    author_results[f'{method}_false_negative'] = cm[1][0]
                    author_results[f'{method}_true_negative'] = cm[1][1]
            
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
                    llm_pure_results = llm_classifier(
                        test_docs,
                        selected_words_info=None,  # No PHC words
                        model=args.model
                    )
                    
                    llm_pure_accuracy = accuracy_score(test_true_labels, llm_pure_results['predictions'])
                    llm_pure_cm = confusion_matrix(test_true_labels, llm_pure_results['predictions'])
                    
                    # Store pure LLM results
                    author_results['LLM_Pure_accuracy'] = llm_pure_accuracy
                    author_results['LLM_Pure_true_positive'] = llm_pure_cm[0][0]
                    author_results['LLM_Pure_false_positive'] = llm_pure_cm[0][1]
                    author_results['LLM_Pure_false_negative'] = llm_pure_cm[1][0]
                    author_results['LLM_Pure_true_negative'] = llm_pure_cm[1][1]
                    
                except Exception as e:
                    print(f"Error in LLM classification: {str(e)}")
                    traceback.print_exc()
            
            # Append results for this author
            results_data.append(author_results)
            
            # Save intermediate results after each author
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(args.output, index=False)
            print(f"\nIntermediate results saved to {args.output}")
            
        except Exception as e:
            print(f"Error processing author {author}: {str(e)}")
            traceback.print_exc()
            continue
    
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