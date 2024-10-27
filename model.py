import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from topicSCORE import topic_score
import re
from typing import Tuple, Dict
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
def preprocess_text(text: str) -> str:
    """
    Preprocess text by converting to lowercase and removing special characters.
    
    Parameters:
    -----------
    text : str
        Input text to preprocess
        
    Returns:
    --------
    str
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def create_vocabulary(df: pd.DataFrame, human_col: str = 'human', 
                     ai_col: str = 'ChatGPT') -> CountVectorizer:
    """
    Create a vocabulary from both human and AI text.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing human and AI text columns
    human_col : str
        Name of the column containing human text
    ai_col : str
        Name of the column containing AI text
        
    Returns:
    --------
    CountVectorizer
        Fitted vectorizer with combined vocabulary
    """
    # Combine all text to create common vocabulary
    all_text = pd.concat([df[human_col], df[ai_col]]).apply(preprocess_text)
    
    # Create and fit vectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(all_text)
    
    return vectorizer

def create_document_word_matrices(df: pd.DataFrame, vectorizer: CountVectorizer, 
                                human_col: str = 'human', 
                                ai_col: str = 'ChatGPT') -> Tuple[csr_matrix, csr_matrix]:
    """
    Create document-word frequency matrices for human and AI text.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing human and AI text columns
    vectorizer : CountVectorizer
        Fitted vectorizer with vocabulary
    human_col : str
        Name of the column containing human text
    ai_col : str
        Name of the column containing AI text
        
    Returns:
    --------
    tuple
        (D_h, D_ai) - Sparse matrices of word frequencies for human and AI text
    """
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

def analyze_text_topics(df: pd.DataFrame, num_topics: int, 
                       human_col: str = 'human', 
                       ai_col: str = 'ChatGPT', 
                       **topic_score_kwargs) -> Dict:
    """
    Analyze topics in human and AI text using Topic SCORE algorithm.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing human and AI text columns
    num_topics : int
        Number of topics to extract
    human_col : str
        Name of the column containing human text
    ai_col : str
        Name of the column containing AI text
    topic_score_kwargs : dict
        Additional arguments to pass to topic_score function
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'A_h': Human topic matrix
        - 'A_ai': AI topic matrix
        - 'vocabulary': List of words in vocabulary
        - 'vectorizer': Fitted CountVectorizer
        - 'D_h': Human document-word frequency matrix
        - 'D_ai': AI document-word frequency matrix
    """
    # Create vocabulary
    vectorizer = create_vocabulary(df, human_col, ai_col)
    
    # Create document-word matrices
    D_h, D_ai = create_document_word_matrices(df, vectorizer, human_col, ai_col)
    # Run topic_score on both matrices
    results_h = topic_score(num_topics, D_h, **topic_score_kwargs)
    results_ai = topic_score(num_topics, D_ai, **topic_score_kwargs)
    
    # Extract word-topic matrices
    A_h = results_h['A_hat']
    A_ai = results_ai['A_hat']
    
    return {
        'A_h': A_h,
        'A_ai': A_ai,
        'vocabulary': vectorizer.get_feature_names_out(),
        'vectorizer': vectorizer,
        'D_h': D_h,
        'D_ai': D_ai
    }

def filter_words(A_h: np.ndarray, A_ai: np.ndarray, t: float, vocabulary: np.ndarray = None) -> dict:
    """
    Filter words based on L1 norm difference between corresponding rows of A_h and A_ai.
    
    Parameters:
    -----------
    A_h : numpy.ndarray
        Human word-topic matrix of shape (p x K)
    A_ai : numpy.ndarray
        AI word-topic matrix of shape (p x K)
    t : float
        Threshold for L1 norm difference
    vocabulary : numpy.ndarray, optional
        Array of vocabulary words corresponding to rows of A_h and A_ai
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'A_hS': Filtered human word-topic matrix
        - 'A_aiS': Filtered AI word-topic matrix
        - 'selected_indices': Indices of selected words
        - 'selected_vocabulary': Selected vocabulary words (if vocabulary provided)
        - 'differences': L1 norm differences for selected words
    """
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

def solve_reconstruction(A: np.ndarray, d: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Solve the optimization problem: min ||Aw - d||_2 subject to ||w||_1 = 1
    with improved convergence handling.
    
    Parameters:
    -----------
    A : numpy.ndarray
        Word-topic matrix (q x K)
    d : numpy.ndarray
        Document vector (q,)
        
    Returns:
    --------
    tuple
        (minimum error, optimal weights)
    """
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
    
    for w0 in initial_points:
        try:
            result = minimize(
                objective,
                w0,
                method='SLSQP',
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

def eval_document(d: np.ndarray, 
                 A_hS: np.ndarray, 
                 A_aiS: np.ndarray, 
                 selected_indices: np.ndarray,
                 return_details: bool = False,
                 debug: bool = False) -> Union[str, Dict]:
    """
    Evaluate a new document and classify it as human or AI-generated.
    
    Parameters:
    -----------
    d : numpy.ndarray
        Document vector of shape (p,) (full vocabulary)
    A_hS : numpy.ndarray
        Filtered human word-topic matrix (q x K)
    A_aiS : numpy.ndarray
        Filtered AI word-topic matrix (q x K)
    selected_indices : numpy.ndarray
        Indices of selected words from the filtering step
    return_details : bool
        If True, return detailed results dictionary
    debug : bool
        If True, print debug information
        
    Returns:
    --------
    str or dict
        Classification result ('human' or 'ChatGPT') or detailed results dictionary
    """
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

def prepare_and_eval_text(text: str, 
                         vectorizer: CountVectorizer,
                         A_hS: np.ndarray, 
                         A_aiS: np.ndarray,
                         selected_indices: np.ndarray,
                         return_details: bool = False) -> Union[str, Dict]:
    """
    Prepare a new text document and evaluate it.
    
    Parameters:
    -----------
    text : str
        Input text to classify
    vectorizer : CountVectorizer
        Fitted vectorizer from training
    A_hS : numpy.ndarray
        Filtered human word-topic matrix
    A_aiS : numpy.ndarray
        Filtered AI word-topic matrix
    selected_indices : numpy.ndarray
        Indices of selected words
    return_details : bool
        If True, return detailed results
        
    Returns:
    --------
    str or dict
        Classification result or detailed results dictionary
    """
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Transform to document vector
    d = vectorizer.transform([processed_text]).toarray()[0]
    
    # Evaluate
    return eval_document(d, A_hS, A_aiS, selected_indices, return_details)




if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Text Classification with Topic SCORE')
    parser.add_argument('--no-cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--K', type=int, default=3, help='Number of topics (if no CV)')
    parser.add_argument('--threshold', type=float, default=0.001, help='Threshold value (if no CV)')
    parser.add_argument('--compare-phc', default=True, help='Compare with PHC method')
    
    args = parser.parse_args()
    
    try:
        # 1. Load data and create initial train/test split
        print("Loading data...")
        df = pd.read_csv('data_cleaned_human_modified.csv')

        # Create initial train/test split
        indices = np.arange(len(df))
        indices_train, indices_test = train_test_split(
            indices, test_size=0.2, random_state=22
        )
        
        df_train = df.iloc[indices_train]
        df_test = df.iloc[indices_test]
        
        print(f"Train size: {len(df_train)}")
        print(f"Test size: {len(df_test)}")
        
        # Create stop words list
        stop_words = list(text.ENGLISH_STOP_WORDS)
        # additional_stops = ['Abstract', 'also', 'using', 'used', 'use', 'one', 'two', 'first', 'second', 
        #                   'paper', 'study', 'method', 'data', 'propose', 'proposed', 
        #                   'show', 'result', 'results', 'find', 'found', 'well', 'may', 
        #                   'might', 'would', 'could']
        additional_stops = ['Abstract']
        stop_words.extend(additional_stops)

        # Create and fit shared vectorizer
        print("\nCreating shared vocabulary...")
        
        # Fit vectorizer on training data only
        train_texts = df_train['human'].tolist() + df_train['ChatGPT'].tolist()
        #vocab = get_vocab(''.join([doc + " " for doc in train_texts]), max_length=10000000)
        #print("vocabbbbbbb = ",len(vocab))
        shared_vectorizer = CountVectorizer(
        stop_words=stop_words,
        min_df=5,
        max_df=0.95,
        token_pattern=r'(?u)\b\w+\b'
        )
        train_texts = [' '.join([word for word in text.split() if len(word) >= 5 and not word.isdigit()]) for text in train_texts]
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
            topic_numbers = [3, 5, 7]
            thresholds = [0.01, 0.005,0.0001,0.00005]
            
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
                            fold_train_texts = df_fold_train['human'].tolist() + df_fold_train['ChatGPT'].tolist()
                            
                            # Transform and normalize training data
                            X_fold_train = shared_vectorizer.transform(fold_train_texts).toarray()
                            doc_lengths = X_fold_train.sum(axis=1)
                            X_fold_train = X_fold_train / doc_lengths[:, np.newaxis]
                            X_fold_train = X_fold_train.T
                            
                            # Split into human and ChatGPT matrices
                            n_fold_train = len(df_fold_train)
                            D_h = X_fold_train[:,:n_fold_train]
                            D_ai = X_fold_train[:,n_fold_train:]
                            
                            # Run TopicScore
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
                                val_matrix = shared_vectorizer.transform(df_fold_val[source]).toarray()
                                val_lengths = val_matrix.sum(axis=1)
                                val_matrix = val_matrix / val_lengths[:, np.newaxis]
                                
                                for doc in val_matrix:
                                    result = eval_document(doc, A_hS, A_aiS, selected_indices)
                                    val_preds.append(result)
                            
                            val_true = ['human'] * len(df_fold_val) + ['ChatGPT'] * len(df_fold_val)
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
                cv_df.to_csv('cv_results.csv', index=False)
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
        
        # Train TopicScore model
        results_h = topic_score(int(best_K), D_h)
        results_ai = topic_score(int(best_K), D_ai)
        
        A_h = results_h['A_hat']
        A_ai = results_ai['A_hat']
        
        print("\nTop words in Human Topics:")
        print("=" * 50)
        print_top_words_per_topic(A_h, vocabulary)

        print("\nTop words in ChatGPT Topics:")
        print("=" * 50)
        print_top_words_per_topic(A_ai, vocabulary)
        
        # Filter words using best threshold
        filtered_results = filter_words(A_h, A_ai, best_threshold, vocabulary)
        topic_score_words = set([w for w, _ in filtered_results['sorted_words']])
        print(f"\nTopic SCORE unique words used: {len(topic_score_words)}")
        
        A_hS = filtered_results['A_hS']
        A_aiS = filtered_results['A_aiS']
        selected_indices = filtered_results['selected_indices']
        
        if 'sorted_words' in filtered_results:
            print("\nTop discriminative words from Topic SCORE:")
            print("=" * 50)
            print(f"{'Word':<20} {'Difference':>12}")
            print("-" * 35)
            for word, diff in filtered_results['sorted_words'][:20]:
                print(f"{word:<20} {diff:>12.4f}")
        
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
                result = eval_document(doc, A_hS, A_aiS, selected_indices, return_details=True)
                test_predictions.append(result['classification'])
                test_details.append(result)
        
        # Calculate TopicScore metrics
        test_true_labels = ['human'] * len(df_test) + ['ChatGPT'] * len(df_test)
        topic_score_accuracy = accuracy_score(test_true_labels, test_predictions)
        topic_score_cm = confusion_matrix(test_true_labels, test_predictions)
        
        print("\nTopic SCORE Results:")
        print(f"Accuracy: {topic_score_accuracy:.4f}")
        print("\nConfusion Matrix:")
        print("                 Predicted Human  Predicted ChatGPT")
        print(f"Actual Human      {topic_score_cm[0][0]:>14d}  {topic_score_cm[0][1]:>17d}")
        print(f"Actual ChatGPT    {topic_score_cm[1][0]:>14d}  {topic_score_cm[1][1]:>17d}")
        print("\nDetailed Classification Report:")
        print(classification_report(test_true_labels, test_predictions))
        
        # Compare with PHC
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
            
            # # Clean data and get PHC predictions
            print("Cleaning data for PHC...")
            phc_train_clean = phc_clean_text(phc_train)
            phc_test_clean = phc_clean_text(phc_test)
            print(f"Training set shape: {phc_train_clean.shape}")
            print(f"Test set shape: {phc_test_clean.shape}")
            
            # # Run PHC with shared vocabulary
            print("Running PHC classification...")
            phc_preds, phc_true = phc_get_pvals(
                phc_train_clean, 
                phc_test_clean, 
                show_hist=False,
                vocab=vocabulary
            )
            # Run PHC with shared vocabulary
          
            print(phc_preds)
            print("true = ",phc_true)
            phc_accuracy = phc_accuracy(phc_preds, phc_true)
            print("\nPHC Results:")
            print(f"Accuracy: {phc_accuracy:.4f}")
            
            # # Calculate PHC metrics
            # phc_cm = confusion_matrix(phc_true, phc_preds)
            # print("\nConfusion Matrix:")
            # print("                 Predicted Human  Predicted ChatGPT")
            # print(f"Actual Human      {phc_cm[0][0]:>14d}  {phc_cm[0][1]:>17d}")
            # print(f"Actual ChatGPT    {phc_cm[1][0]:>14d}  {phc_cm[1][1]:>17d}")
            # print("\nDetailed Classification Report:")
            # print(classification_report(phc_true, phc_preds))
            
            # # Compare selected words
            # phc_words = set([word_info['word'] for word_info in phc_words_info])
            # common_words = topic_score_words.intersection(phc_words)
            
            # print("\nWord Selection Comparison:")
            # print(f"TopicScore selected words: {len(topic_score_words)}")
            # print(f"PHC selected words: {len(phc_words)}")
            # print(f"Words selected by both methods: {len(common_words)}")
            
            # if len(common_words) > 0:
            #     print("\nTop 20 common words between methods:")
            #     print("=" * 40)
            #     for word in sorted(list(common_words))[:20]:
            #         print(word)
            
            # # Save results
            # print("\nSaving results...")
            
            # # Method comparison
            # comparison_df = pd.DataFrame({
            #     'Method': ['TopicScore', 'PHC'],
            #     'Accuracy': [topic_score_accuracy, phc_accuracy],
            #     'Selected_Words': [len(topic_score_words), len(phc_words)],
            #     'Common_Words': [len(common_words), len(common_words)]
            # })
            # comparison_df.to_csv('method_comparison.csv', index=False)
            
            # # Word selections
            # words_df = pd.DataFrame({
            #     'TopicScore_Words': pd.Series([w for w, _ in filtered_results['sorted_words']]),
            #     'TopicScore_Importance': pd.Series([d for _, d in filtered_results['sorted_words']]),
            #     'PHC_Words': pd.Series([w['word'] for w in phc_words_info]),
            #     'PHC_Effect_Size': pd.Series([w['effect_size'] for w in phc_words_info]),
            #     'PHC_Direction': pd.Series([w['direction'] for w in phc_words_info])
            # })
            # words_df.to_csv('word_selections.csv', index=False)
            
            # # Detailed test results
            # test_results_df = pd.DataFrame({
            #     'Text': list(df_test['human']) + list(df_test['ChatGPT']),
            #     'True_Label': test_true_labels,
            #     'TopicScore_Prediction': test_predictions,
            #     'PHC_Prediction': phc_preds,
            #     'Human_Error': [d['error_human'] for d in test_details],
            #     'AI_Error': [d['error_ai'] for d in test_details],
            #     'Error_Margin': [abs(d['error_human'] - d['error_ai']) for d in test_details]
            # })
            # test_results_df.to_csv('test_results.csv', index=False)
            
            # print("\nResults saved to:")
            # print("- method_comparison.csv")
            # print("- word_selections.csv")
            # print("- test_results.csv")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
























# if __name__ == "__main__":
#     # Add argument parsing
#     parser = argparse.ArgumentParser(description='Text Classification with Topic SCORE')
#     parser.add_argument('--cv', 
#                    dest='no_cv',
#                    action='store_false',
#                    help='Perform cross-validation')
#     parser.add_argument('--no-cv',
#                    action='store_true',
#                    help='Skip cross-validation')
#     parser.add_argument('--K', type=int, default=15, help='Number of topics (if no CV)')
#     parser.add_argument('--threshold', type=float, default=0.005, help='Threshold value (if no CV)')
#     parser.add_argument('--compare-phc', default=True, help='Compare with PHC method')
    
#     args = parser.parse_args()
#     print(args.no_cv)
#     try:
#         # 1. Load data and combine human and ChatGPT texts with labels
#         print("Loading data...")
#         df = pd.read_csv('data_cleaned_human_modified.csv')
        
#         # Combine all texts and their corresponding labels
#         all_texts = []
#         all_labels = []
        
#         # Add human texts
#         all_texts.extend(df['human'].tolist())
#         all_labels.extend(['human'] * len(df))
        
#         # Add ChatGPT texts
#         all_texts.extend(df['ChatGPT'].tolist())
#         all_labels.extend(['ChatGPT'] * len(df))
        
#         # Create a combined DataFrame
#         combined_df = pd.DataFrame({
#             'text': all_texts,
#             'label': all_labels
#         })
        
#         # Create random train/test split
#         texts_train, texts_test, labels_train, labels_test = train_test_split(
#             combined_df['text'],
#             combined_df['label'],
#             test_size=0.3,
#             random_state=33,
#             stratify=combined_df['label']
#         )
        
#         print(f"Train size: {len(texts_train)}")
#         print(f"Test size: {len(texts_test)}")
        
#         # Create stop words list
#         stop_words = set(word.lower() for word in text.ENGLISH_STOP_WORDS)
#         additional_stops = {'abstract', 'the', 'of', 'and', 'a', 'in', 'to', 'for', 
#                        'is', 'that', 'we', 'are', 'on', 'with', 'by', 'our', 
#                        'this', 'it', 'its', 'be', 'as', 'at', 'which', 'from'}
#         stop_words.update(additional_stops)
    

#         # Create and fit shared vectorizer
#         print("\nCreating shared vocabulary...")
#         train_texts = texts_train.tolist()
#         #vocab = get_vocab(''.join([doc + " " for doc in train_texts]))
#         shared_vectorizer = CountVectorizer(
#             stop_words=list(stop_words),
#             min_df=5,
#             max_df=0.95,
#             token_pattern=r'(?u)\b\w+\b'
#         )
        
#         train_texts = [' '.join([word for word in text.split() if len(word.strip()) >= 3]) for text in train_texts]
#         shared_vectorizer.fit(train_texts)
#         vocabulary = shared_vectorizer.get_feature_names_out()
#         print(f"Vocabulary size: {len(vocabulary)}")
#         print(vocabulary[:20])

#         # Parameter selection
#         if args.no_cv:
#             print("helloooo")
#             print(f"\nSkipping CV, using specified parameters: K={args.K}, threshold={args.threshold}")
#             best_K = args.K
#             best_threshold = args.threshold
#         else:
#             print("\nPerforming cross-validation for parameter selection...")
            
#             # Create train/val split from training data
#             # Create train/val split from training data
#             kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
#             # Define parameter grid
#             topic_numbers = [5, 10, 15, 20, 25, 30]
#             thresholds = [0.1,0.05, 0.01,0.005,0.002]
            
#             cv_results = []
#             best_score = 0
#             best_K = None
#             best_threshold = None
            
#             for K in topic_numbers:
#                 for threshold in thresholds:
#                     print(f"\nTesting K={K}, threshold={threshold}")
#                     fold_scores = []
                    
#                     # Store texts and labels as numpy arrays for easier indexing
#                     texts_array = np.array(train_texts)
#                     labels_array = np.array(labels_train)
                    
#                     for fold, (train_idx, val_idx) in enumerate(kf.split(texts_array), 1):
#                         try:
#                             # Split training data into train/val for this fold
#                             texts_fold_train = texts_array[train_idx]
#                             print('so far so good')
#                             texts_fold_val = texts_array[val_idx]
#                             labels_fold_train = labels_array[train_idx]
#                             labels_fold_val = labels_array[val_idx]
                            
#                             # Transform and normalize fold training data
#                             X_fold_train = shared_vectorizer.transform(texts_fold_train).toarray()
#                             doc_lengths = X_fold_train.sum(axis=1)
#                             X_fold_train = X_fold_train / doc_lengths[:, np.newaxis]
#                             X_fold_train = X_fold_train.T
#                             #print("NOT TOOO BAD")
#                             # Split into human and ChatGPT matrices
#                             human_mask = labels_fold_train == 'human'
#                             if not any(human_mask) or not any(~human_mask):
#                                 print(f"Fold {fold}: Skipping due to imbalanced split")
#                                 continue
                                
#                             D_h = X_fold_train[:, human_mask]
#                             D_ai = X_fold_train[:, ~human_mask]
                            
#                             if D_h.shape[1] == 0 or D_ai.shape[1] == 0:
#                                 print(f"Fold {fold}: Empty matrices detected, skipping...")
#                                 continue
                            
#                             # Run TopicScore with error handling
#                             try:
#                                 results_h = topic_score(K, D_h)
#                                 #PRINT("DONENENENENE")
#                                 results_ai = topic_score(K, D_ai)
                                
#                                 A_h = results_h['A_hat']
#                                 A_ai = results_ai['A_hat']
#                             except Exception as e:
#                                 print(f"Fold {fold}: Topic SCORE failed - {str(e)}")
#                                 continue
                            
#                             # Filter words
#                             filtered_results = filter_words(A_h, A_ai, threshold, vocabulary)
                            
#                             if len(filtered_results['selected_indices']) == 0:
#                                 print(f"Fold {fold}: No words selected, skipping...")
#                                 continue
                            
#                             A_hS = filtered_results['A_hS']
#                             A_aiS = filtered_results['A_aiS']
#                             selected_indices = filtered_results['selected_indices']
                            
#                             # Transform and normalize validation data
#                             X_fold_val = shared_vectorizer.transform(texts_fold_val).toarray()
#                             val_lengths = X_fold_val.sum(axis=1)
#                             val_mask = val_lengths != 0
#                             X_fold_val[val_mask] = X_fold_val[val_mask] / val_lengths[val_mask, np.newaxis]
                            
#                             # Validation predictions
#                             val_preds = []
#                             for doc in X_fold_val:
#                                 try:
#                                     result = eval_document(doc, A_hS, A_aiS, selected_indices)
#                                     val_preds.append(result)
#                                 except Exception as e:
#                                     print(f"Error in document evaluation: {str(e)}")
#                                     continue
                            
#                             if len(val_preds) != len(labels_fold_val):
#                                 print(f"Fold {fold}: Prediction count mismatch, skipping...")
#                                 continue
                                
#                             fold_acc = accuracy_score(labels_fold_val, val_preds)
#                             fold_scores.append(fold_acc)
#                             print(f"Fold {fold} accuracy: {fold_acc:.4f}")
                            
#                         except Exception as e:
#                             print(f"Error in fold {fold}: {str(e)}")
#                             traceback.print_exc()
#                             continue
                    
#                     if fold_scores:
#                         avg_score = np.mean(fold_scores)
#                         std_score = np.std(fold_scores)
#                         cv_results.append({
#                             'K': K,
#                             'threshold': threshold,
#                             'mean_accuracy': avg_score,
#                             'std_accuracy': std_score,
#                             'n_folds': len(fold_scores)
#                         })
#                         print(f"Average accuracy: {avg_score:.4f} (±{std_score:.4f})")
                        
#                         if avg_score > best_score:
#                             best_score = avg_score
#                             best_K = K
#                             best_threshold = threshold

#         # Train final model using best parameters
#         print(f"\nTraining final model with K={best_K}, threshold={best_threshold}...")
        
#         # Transform and normalize all training data
#         X_train = shared_vectorizer.transform(train_texts).toarray()
#         doc_lengths = X_train.sum(axis=1)
#         X_train = X_train / doc_lengths[:, np.newaxis]
#         X_train = X_train.T
        
#         # Split into human and ChatGPT matrices using labels
#         human_mask = labels_train == 'human'
#         D_h = X_train[:, human_mask]
#         D_ai = X_train[:, ~human_mask]
        
#         # Train TopicScore model
#         results_h = topic_score(int(best_K), D_h)
#         results_ai = topic_score(int(best_K), D_ai)
        
#         A_h = results_h['A_hat']
#         A_ai = results_ai['A_hat']
        
#         print("\nTop words in Human Topics:")
#         print("=" * 50)
#         print_top_words_per_topic(A_h, vocabulary)

#         print("\nTop words in ChatGPT Topics:")
#         print("=" * 50)
#         print_top_words_per_topic(A_ai, vocabulary)
        
#         # Filter words using best threshold
#         filtered_results = filter_words(A_h, A_ai, best_threshold, vocabulary)
#         topic_score_words = set([w for w, _ in filtered_results['sorted_words']])
#         print(f"\nTopic SCORE unique words used: {len(topic_score_words)}")
        
#         A_hS = filtered_results['A_hS']
#         A_aiS = filtered_results['A_aiS']
#         selected_indices = filtered_results['selected_indices']
        
#         if 'sorted_words' in filtered_results:
#             print("\nTop discriminative words from Topic SCORE:")
#             print("=" * 50)
#             print(f"{'Word':<20} {'Difference':>12}")
#             print("-" * 35)
#             for word, diff in filtered_results['sorted_words'][:20]:
#                 print(f"{word:<20} {diff:>12.4f}")
        
#         # Evaluate on test set
#         print("\nEvaluating on test set...")
#         test_predictions = []
#         test_details = []
        
#         # Process test documents
#         test_matrix = shared_vectorizer.transform(texts_test).toarray()
#         test_lengths = test_matrix.sum(axis=1)
#         test_matrix = test_matrix / test_lengths[:, np.newaxis]
        
#         for doc in test_matrix:
#             result = eval_document(doc, A_hS, A_aiS, selected_indices, return_details=True)
#             test_predictions.append(result['classification'])
#             test_details.append(result)
        
#         # Calculate TopicScore metrics
#         topic_score_accuracy = accuracy_score(labels_test, test_predictions)
#         topic_score_cm = confusion_matrix(labels_test, test_predictions)
        
#         print("\nTopic SCORE Results:")
#         print(f"Accuracy: {topic_score_accuracy:.4f}")
#         print("\nConfusion Matrix:")
#         print("                 Predicted Human  Predicted ChatGPT")
#         print(f"Actual Human      {topic_score_cm[0][0]:>14d}  {topic_score_cm[0][1]:>17d}")
#         print(f"Actual ChatGPT    {topic_score_cm[1][0]:>14d}  {topic_score_cm[1][1]:>17d}")
#         print("\nDetailed Classification Report:")
#         print(classification_report(labels_test, test_predictions))
        
#         # Compare with PHC
#         if args.compare_phc:
#             print("\nComparing with PHC method...")
            
#             # Prepare data for PHC
#             phc_train = pd.DataFrame({
#                 'author': ['author1' if label == 'human' else 'author2' for label in labels_train],
#                 'title': [''] * len(texts_train),
#                 'text': texts_train
#             })
            
#             phc_test = pd.DataFrame({
#                 'author': ['author1' if label == 'human' else 'author2' for label in labels_test],
#                 'title': [''] * len(texts_test),
#                 'text': texts_test
#             })
            
#             # Clean data and get PHC predictions
#             print("Cleaning data for PHC...")
#             phc_train_clean = phc_clean_text(phc_train)
#             phc_test_clean = phc_clean_text(phc_test)
#             print(f"Training set shape: {phc_train_clean.shape}")
#             print(f"Test set shape: {phc_test_clean.shape}")
            
#             # Run PHC with shared vocabulary
#             print("Running PHC classification...")
#             phc_preds, phc_true = phc_get_pvals(
#                 phc_train_clean, 
#                 phc_test_clean, 
#                 show_hist=False,
#                 vocab=vocabulary
#             )
            
#             print(phc_preds)
#             print("true = ", phc_true)
#             phc_accuracy = phc_accuracy(phc_preds, phc_true)
#             print("\nPHC Results:")
#             print(f"Accuracy: {phc_accuracy:.4f}")

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         traceback.print_exc()