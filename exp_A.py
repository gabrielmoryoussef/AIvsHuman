import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text as text_
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import warnings
import itertools
import argparse
from typing import Dict, List, Tuple, Union
import json
from scipy.stats import norm
import re
from collections import Counter
import heapq
from openai import OpenAI
import time
import os
import nltk
from nltk.tokenize import sent_tokenize
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from gpt2model import GPT2PPLV2
# nltk.data.path.append(os.path.expanduser('~/nltk_data'))
# nltk.download('punkt')

class TextDataLoader:
    """Base class for loading different text dataset formats"""
    def __init__(self):
        self.data = None
        self.labels = None
    
    def load_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Should return a DataFrame with 'text' and 'label' columns"""
        raise NotImplementedError
        
    def get_label_pairs(self) -> List[Tuple[str, str]]:
        """Returns all possible label pairs for binary classification"""
        unique_labels = list(set(self.labels))
        return [(unique_labels[i], unique_labels[j]) 
                for i in range(len(unique_labels)) 
                for j in range(i + 1, len(unique_labels))]
    
class CoAuthorDataLoader(TextDataLoader):
    def __init__(self, dataset_dir: str):
        super().__init__()
        self.dataset_dir = dataset_dir
    
    def _split_into_sentences(self, text):
        """Split text into sentences using sent_tokenize"""
        try:
            return sent_tokenize(text)
        except Exception as e:
            print(f"Error in sentence tokenization: {e}")
            # Fallback to simple splitting
            sentences = text.split('.')
            return [s.strip() + '.' for s in sentences if s.strip()]

    def _find_writing_sessions(self):
        """Find all JSONL files in the dataset directory"""
        return [
            os.path.join(self.dataset_dir, path)
            for path in os.listdir(self.dataset_dir) 
            if path.endswith('.jsonl')
        ]
    
    def _apply_ops(self, doc, mask, ops, source):
        """Apply operations to track changes in text"""
        original_doc = doc
        original_mask = mask

        new_doc = ''
        new_mask = ''
        for op in ops:
            # Handle retain operation
            if 'retain' in op:
                num_char = op['retain']
                retain_doc = original_doc[:num_char]
                retain_mask = original_mask[:num_char]
                original_doc = original_doc[num_char:]
                original_mask = original_mask[num_char:]
                new_doc = new_doc + retain_doc
                new_mask = new_mask + retain_mask

            # Handle insert operation
            elif 'insert' in op:
                insert_doc = op['insert']
                insert_mask = 'U' * len(insert_doc)  # User
                if source == 'api':
                    insert_mask = 'A' * len(insert_doc)  # API

                if isinstance(insert_doc, dict):
                    if 'image' in insert_doc:
                        print('Skipping invalid object insertion (image)')
                    else:
                        print('Ignore invalid insertions:', op)
                else:
                    new_doc = new_doc + insert_doc
                    new_mask = new_mask + insert_mask

            # Handle delete operation
            elif 'delete' in op:
                num_char = op['delete']
                if original_doc:
                    original_doc = original_doc[num_char:]
                    original_mask = original_mask[num_char:]
                else:
                    new_doc = new_doc[:-num_char]
                    new_mask = new_mask[:-num_char]

        final_doc = new_doc + original_doc
        final_mask = new_mask + original_mask
        return final_doc, final_mask

    def _get_text_and_mask(self, events, event_id, remove_prompt=True):
        """Get text and mask for events up to event_id"""
        prompt = events[0]['currentDoc'].strip()
        
        text = prompt
        mask = 'P' * len(prompt)  # Prompt
        
        for event in events[:event_id]:
            if 'textDelta' not in event or 'ops' not in event['textDelta']:
                continue
            ops = event['textDelta']['ops']
            source = event['eventSource']
            text, mask = self._apply_ops(text, mask, ops, source)

        if remove_prompt and 'P' in mask:
            end_index = mask.rindex('P')
            text = text[end_index + 1:]
            mask = mask[end_index + 1:]

        return text, mask
    
    def _identify_author(self, mask):
        """Identify author based on mask content"""
        if 'P' in mask:
            return 'prompt'
        elif 'U' in mask and 'A' in mask:
            return 'both'
        elif 'U' in mask and 'A' not in mask:
            return 'human'
        elif 'U' not in mask and 'A' in mask:
            return 'ChatGPT'
        else:
            raise RuntimeError(f'Could not identify author for mask: {mask}')
    
    def load_data(self) -> Tuple[pd.DataFrame, List[str]]:
        session_paths = self._find_writing_sessions()
        print(f"Found {len(session_paths)} writing sessions")
        
        all_texts = []
        all_labels = []
        
        for session_path in session_paths:
            try:
                with open(session_path, 'r') as f:
                    events = [json.loads(line) for line in f]
                
                text, mask = self._get_text_and_mask(events, len(events), remove_prompt=True)
                sentences = self._split_into_sentences(text.strip())
                
                for sentence in sentences:
                    if sentence not in text:
                        continue
                    
                    index = text.index(sentence)
                    sentence_mask = mask[index:index + len(sentence)]
                    author = self._identify_author(sentence_mask)
                    
                    if author in ['human', 'ChatGPT']:
                        all_texts.append(sentence)
                        all_labels.append(author)
                    
            except Exception as e:
                print(f"Error processing session {session_path}: {str(e)}")
                continue
        
        if not all_texts:
            raise ValueError("No valid sentences found in any session file")
            
        self.data = pd.DataFrame({
            'text': all_texts,
            'label': all_labels
        })
        
        print("\nDataset Statistics:")
        print(f"Total samples: {len(self.data)}")
        print("Label distribution:")
        print(self.data['label'].value_counts())
        
        self.labels = all_labels
        return self.data, self.labels
    
    def get_label_pairs(self) -> List[Tuple[str, str]]:
        return [('human', 'ChatGPT')]

# class CoAuthorDataLoader(TextDataLoader):
#     """Loader for CoAuthor dataset format"""
#     def __init__(self, events_path: str):
#         super().__init__()
#         self.events_path = events_path
        
#     def load_data(self) -> Tuple[pd.DataFrame, List[str]]:
#         # Read events from jsonl file
#         events = []
#         with open(self.events_path, 'r') as f:
#             for line in f:
#                 events.append(json.loads(line))
                
#         # Extract text and source info with masks
#         texts = []
#         labels = []
#         masks = []
        
#         current_doc = ''
#         current_mask = ''
        
#         for event in events:
#             if 'textDelta' in event and 'ops' in event['textDelta']:
#                 # Apply operations to update document
#                 ops = event['textDelta']['ops']
#                 source = event['eventSource']  # 'api' or 'user'
                
#                 # Update document and mask
#                 current_doc, current_mask = apply_ops(
#                     current_doc, 
#                     current_mask, 
#                     ops, 
#                     source
#                 )
                
#                 # Store if document has content
#                 if current_doc.strip():
#                     texts.append(current_doc)
#                     labels.append(source)
#                     masks.append(current_mask)
        
#         self.data = pd.DataFrame({
#             'text': texts,
#             'label': labels,
#             'mask': masks
#         })
#         self.labels = labels
        
#         return self.data, self.labels

class StandardCSVDataLoader(TextDataLoader):
    """Loader for standard CSV format with human/AI columns"""
    def __init__(self, csv_path: str, human_col: str = 'human', ai_col: str = 'ChatGPT'):
        super().__init__()
        self.csv_path = csv_path
        self.human_col = human_col
        self.ai_col = ai_col
        
    def load_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load data from CSV file"""
        print(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        # Reshape data to long format
        texts = list(df[self.human_col]) + list(df[self.ai_col])
        self.labels = ['human'] * len(df) + ['ChatGPT'] * len(df)
        
        self.data = pd.DataFrame({
            'text': texts,
            'label': self.labels
        })
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Labels distribution: {pd.Series(self.labels).value_counts()}")
        
        return self.data, self.labels

    def get_label_pairs(self) -> List[Tuple[str, str]]:
        """Get all possible label pairs for binary classification"""
        if self.labels is None:
            self.load_data()
        return [('human', 'ChatGPT')]

class TextClassifier:
    """Base class for text classification methods"""
    def __init__(self, vectorizer_params: Dict = None):
        self.vectorizer_params = vectorizer_params or {
            'min_df': 5,
            'max_df': 0.95,
            'stop_words': 'english'
        }
        self.vectorizer = CountVectorizer(**self.vectorizer_params)
        
    def preprocess(self, text: str) -> str:
        # Basic preprocessing - can be extended
        text = text.lower()
        return text
        
    def fit(self, X_train: pd.DataFrame, y_train: List[str]):
        raise NotImplementedError
        
    def predict(self, X_test: pd.DataFrame) -> List[str]:
        raise NotImplementedError


class EmbeddingsRandomForestClassifier:
    """Sentence Transformer Embeddings + Random Forest classifier implementation"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 n_estimators: int = 100, 
                 batch_size: int = 32):
        """
        Initialize the classifier
        
        Args:
            model_name: Name of the sentence transformer model to use
            n_estimators: Number of trees in random forest
            batch_size: Batch size for computing embeddings
        """
        self.model_name = model_name
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.st_model = SentenceTransformer(model_name)
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        
    def fit(self, X_train: pd.Series, y_train: List[str]):
        """
        Fit the classifier
        
        Args:
            X_train: Training texts
            y_train: Training labels ('human' or 'ChatGPT')
        """
        print("Computing embeddings for training data...")
        train_embeddings = self.st_model.encode(
            X_train.tolist(),
            batch_size=self.batch_size
        )
        
        print("Training Random Forest...")
        self.rf_model.fit(train_embeddings, y_train)
        return self
        
    def predict(self, X_test: pd.Series) -> np.ndarray:
        """
        Make predictions on test data
        
        Args:
            X_test: Test texts
            
        Returns:
            Array of predictions ('human' or 'ChatGPT')
        """
        print("Computing embeddings for test data...")
        test_embeddings = self.st_model.encode(
            X_test.tolist(),
            batch_size=self.batch_size
        )
        
        return self.rf_model.predict(test_embeddings)

class TopicScoreClassifier(TextClassifier):
    """Implementation of TopicScore classification method"""
    def __init__(self, n_topics: int = 3, share_weights=False,threshold: float = 0.0001, **kwargs):
        super().__init__(**kwargs)
        self.n_topics = n_topics
        self.threshold = threshold
        self.A_h = None
        self.A_ai = None
        self.selected_indices = None
        self.vocabulary = None
        self.share_weights = share_weights
        
    def topic_score(self, K, X, K0=None, m=None, Mquantile=0, num_restart=1, seed=None):
        """
        Topic SCORE algorithm with guaranteed probability distributions.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if K0 is None:
            K0 = int(np.ceil(1.5 * K))
        if m is None:
            m = 10*K
        
        p, n = X.shape
        epsilon = 1e-10
        
        # Compute row means and handle truncation
        M = np.mean(X, axis=1)
        M_trunk = np.minimum(M, np.quantile(M, Mquantile)) if Mquantile > 0 else M
        M_trunk = M_trunk + epsilon
        
        # SVD step
        X_scaled = np.diag(np.sqrt(1/M_trunk)) @ X
        U, _, _ = svds(X_scaled, K)
        Xi = np.fliplr(U)
        
        # Calculate ratios
        Xi[:, 0] = np.abs(Xi[:, 0])
        mask = Xi[:, 0] > epsilon
        R = np.zeros((p, K-1))
        R[mask] = (Xi[mask, 1:K] / Xi[mask, 0:1]).reshape(-1, K-1)
        R = np.nan_to_num(R, 0)
        
        # Find vertices
        V, theta = self._vertices_est(R, K0, m, num_restart)
        
        # Calculate convex combinations
        augmented_R = np.column_stack([R, np.ones(p)])
        augmented_V = np.column_stack([V, np.ones(K)])
        
        try:
            Pi = augmented_R @ np.linalg.inv(augmented_V)
        except np.linalg.LinAlgError:
            Pi = augmented_R @ np.linalg.pinv(augmented_V)
        
        # Ensure non-negative values and normalize
        Pi = np.maximum(Pi, 0)
        Pi = Pi / (Pi.sum(axis=1, keepdims=True) + epsilon)
        
        # Calculate word-topic matrix
        A_hat = np.sqrt(M_trunk)[:, np.newaxis] * Xi[:, 0:1] * Pi
        
        # Normalize columns to sum to 1
        col_sums = A_hat.sum(axis=0)
        for j in range(K):
            if col_sums[j] < epsilon:
                A_hat[:, j] = 1/p
            else:
                A_hat[:, j] = A_hat[:, j] / col_sums[j]
        
        return {'A_hat': A_hat, 'R': R, 'V': V, 'Pi': Pi, 'theta': theta}
    
    def _vertices_est(self, R, K0, m, num_restart):
        K = R.shape[1] + 1
        
        # KMeans clustering
        m = min(m, 5*K)
        kmeans = KMeans(n_clusters=m, n_init=num_restart)
        kmeans.fit(R)
        theta = kmeans.cluster_centers_
        theta_original = theta.copy()
        
        # Find initial vertices
        inner = theta @ theta.T
        distance = np.diag(inner)[:, np.newaxis] + np.diag(inner) - 2 * inner
        top2_idx = np.unravel_index(np.argmax(distance), distance.shape)
        theta0 = theta[list(top2_idx)]
        mask = np.ones(len(theta), dtype=bool)
        mask[list(top2_idx)] = False
        theta = theta[mask]
        
        # Greedy selection
        if K0 > 2:
            for _ in range(K-2):
                if len(theta) == 0:
                    break
                inner = theta @ theta.T
                distance = np.ones((len(theta0), 1)) @ np.diag(inner)[np.newaxis, :] - 2 * theta0 @ theta.T
                ave_dist = np.mean(distance, axis=0)
                idx = np.argmax(ave_dist)
                theta0 = np.vstack([theta0, theta[idx]])
                theta = np.delete(theta, idx, axis=0)
        
        return theta0[:K], theta_original
    
    def _filter_words(self, A_h, A_ai, vocabulary=None):
        # Compute L1 norm of differences for each word
        differences = np.linalg.norm(A_h - A_ai, ord=1, axis=1)
        
        # Find indices where difference exceeds threshold
        selected_indices = np.where(differences > self.threshold)[0]
        
        # Create filtered matrices
        A_hS = A_h[selected_indices, :]
        A_aiS = A_ai[selected_indices, :]
        
        result = {
            'A_hS': A_hS,
            'A_aiS': A_aiS,
            'selected_indices': selected_indices,
            'differences': differences[selected_indices]
        }
        
        if vocabulary is not None:
            result['selected_vocabulary'] = vocabulary[selected_indices]
            
            # Create sorted version by difference magnitude
            sorted_indices = np.argsort(-differences[selected_indices])
            result['sorted_words'] = [
                (vocabulary[selected_indices[i]], differences[selected_indices[i]])
                for i in sorted_indices
            ]
            print(f"Using {len(result['sorted_words'])} with T-SCORE")
            print("T-SCORE words: ", result['sorted_words'][:20])
        return result
        
    def _solve_reconstruction(self, A, d):
        K = A.shape[1]
        epsilon = 1e-10
        
        def objective(w):
            return np.linalg.norm(A @ w - d)**2
        
        def l1_constraint(w):
            return np.sum(np.abs(w)) - 1
        
        best_result = None
        best_obj = float('inf')
        
        initial_points = [
            np.ones(K) / K,
            np.random.dirichlet(np.ones(K)),
            np.zeros(K)
        ]
        initial_points[-1][0] = 1.0
        
        constraint = {'type': 'eq', 'fun': l1_constraint}
        
        for w0 in initial_points:
            try:
                result = minimize(
                    objective,
                    w0,
                    method='SLSQP',
                    constraints=[constraint],
                    bounds=[(0, None) for _ in range(K)],
                    options={'ftol': 1e-6, 'maxiter': 5000, 'disp': False, 'eps': 1e-8}
                )
                
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result
                    
            except Exception:
                continue
        
        if best_result is None:
            w = np.ones(K) / K
            return objective(w), w
        
        w = np.maximum(best_result.x, 0)
        w = w / (np.sum(w) + epsilon)
        
        return objective(w), w
    
    def fit(self, X_train: pd.DataFrame, y_train: List[str]):
        """Fit the TopicScore classifier"""
        # Transform texts to document-term matrix
        X = self.vectorizer.fit_transform(X_train)
        self.vocabulary = self.vectorizer.get_feature_names_out()
        
        # Split into human and AI matrices
        mask_human = np.array(y_train) == 'human'
        X_human = X[mask_human].toarray().T
        X_ai = X[~mask_human].toarray().T
        
        # Normalize matrices
        X_human = X_human / (X_human.sum(axis=0) + 1e-10)
        X_ai = X_ai / (X_ai.sum(axis=0) + 1e-10)
        
        if self.share_weights:
            # Stack matrices vertically and normalize by 2
            D_combined = np.vstack([X_human, X_ai]) / 2
            results_combined = self.topic_score(self.n_topics, D_combined)
            
            # Split the result back into human and AI matrices
            p = X_human.shape[0]  # number of words
            A_combined = results_combined['A_hat']
            self.A_h = A_combined[:p, :]
            self.A_ai = A_combined[p:, :]
        else:
            # Run TopicScore separately on each matrix
            results_h = self.topic_score(self.n_topics, X_human)
            results_ai = self.topic_score(self.n_topics, X_ai)
            self.A_h = results_h['A_hat']
            self.A_ai = results_ai['A_hat']
        
        # Filter words
        filtered_results = self._filter_words(self.A_h, self.A_ai, self.vocabulary)
        self.A_h = filtered_results['A_hS']
        self.A_ai = filtered_results['A_aiS']
        self.selected_indices = filtered_results['selected_indices']
        
        return self
        
    def predict(self, X_test: pd.DataFrame) -> List[str]:
        """Predict using the TopicScore classifier"""
        # Transform test texts
        X = self.vectorizer.transform(X_test)
        predictions = []
        
        # For each document
        for doc in X:
            doc_vector = doc.toarray()[0]
            
            # Normalize document vector
            if np.sum(doc_vector) != 0:
                doc_vector = doc_vector / np.sum(doc_vector)
            
            # Filter to selected words
            d_S = doc_vector[self.selected_indices]
            
            # Compute reconstruction errors
            error_human, _ = self._solve_reconstruction(self.A_h, d_S)
            error_ai, _ = self._solve_reconstruction(self.A_ai, d_S)
            
            # Classify based on minimum reconstruction error
            predictions.append('human' if error_human < error_ai else 'ChatGPT')
        
        return predictions

class TFIDFClassifier(TextClassifier):
    """TF-IDF with Logistic Regression classifier implementation"""
    
    def __init__(self, **kwargs):
        """
        Initialize TF-IDF classifier
        
        Args:
            **kwargs: Arguments to pass to CountVectorizer
        """
        # Update default parameters for TF-IDF
        tfidf_params = {
            'min_df': 5,
            'max_df': 0.95,
            'stop_words': 'english',
            'token_pattern': r'(?u)\b\w+\b'
        }
        tfidf_params.update(kwargs)
        super().__init__(vectorizer_params=tfidf_params)
        
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
    def fit(self, X_train: pd.Series, y_train: pd.Series) -> 'TFIDFClassifier':
        """
        Fit the TF-IDF classifier
        
        Args:
            X_train: Training texts
            y_train: Training labels ('human' or 'ChatGPT')
        """
        # Transform texts to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Train logistic regression
        self.classifier.fit(X_tfidf, y_train)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.vectorizer.get_feature_names_out(),
            'importance': abs(self.classifier.coef_[0])
        }).sort_values('importance', ascending=False)
        
        # Print top discriminative features
        print("\nTop discriminative words:")
        print(self.feature_importance.head(20))
        
        return self
        
    def predict(self, X_test: pd.Series) -> np.ndarray:
        """
        Make predictions using the TF-IDF classifier
        
        Args:
            X_test: Test texts
            
        Returns:
            Array of predictions ('human' or 'ChatGPT')
        """
        # Transform texts and predict
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_tfidf)
class PHCClassifier:
    """Paired Human Classifier implementation"""
    
    def __init__(self, max_vocab_length: int = 1000, gamma: float = 0.25, 
                 threshold: float = 0.99, ef_corr: bool = True):
        self.max_vocab_length = max_vocab_length
        self.gamma = gamma
        self.threshold = threshold
        self.ef_corr = ef_corr
        self.vocabulary = None
        self.selected_words_info = None
        self.vectorizer = None
        
    def _topk_frequent(self, nums: List, k: int) -> List:
        """Get top-k frequent items"""
        dic = Counter(nums)
        heapmax = [[-freq, num] for num, freq in dic.items()]
        heapq.heapify(heapmax)
        return [heapq.heappop(heapmax)[1] for _ in range(min(k, len(heapmax)))]
    
    def _get_vocab(self, text: str) -> List[str]:
        """Extract vocabulary from text"""
        vocab = text.split()
        k = min(self.max_vocab_length, len(set(vocab)))
        return self._topk_frequent(vocab, k)
    
    def _clean_text(self, text: str) -> str:
        """Clean text according to PHC requirements"""
        text = re.sub("All rights", "", text)
        text = re.sub("reserved", "", text)
        text = re.sub("[^A-Za-z ]", "", text)
        text = re.sub("copyright", "", text)
        return text.lower()
    
    def _doc_to_dtm(self, texts: List[str], vocab: List[str]) -> np.ndarray:
        """Convert documents to document-term matrix"""
        stop_words = list(text_.ENGLISH_STOP_WORDS)
        additional_stops = ['Abstract']
        stop_words.extend(additional_stops)
        
        self.vectorizer = CountVectorizer(
            tokenizer=lambda txt: txt.split(),
            vocabulary=vocab,
            stop_words=stop_words
        )
        return self.vectorizer.transform(texts).toarray()
    
    def _estimate_poisson(self, corpus: np.ndarray) -> np.ndarray:
        """Estimate Poisson parameters"""
        return np.mean(corpus, axis=0)
    
    def _higher_criticism(self, pvals: np.ndarray) -> Tuple[float, int]:
        """Compute Higher Criticism statistic"""
        pvals = np.sort(pvals[pvals <= self.threshold])
        N = len(pvals)
        hc = -1000
        i_star = 0
        
        for i in range(1, int(self.gamma * N) + 1):
            if pvals[i-1] >= 1/N:
                num = np.sqrt(N) * ((i/N) - pvals[i-1])
                den = np.sqrt((i/N) * (1-i/N))
                cur = num/den
                if cur > hc:
                    hc = cur
                    i_star = i
        return hc, i_star
    
    def _evaluate(self, new_data: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """Evaluate new data"""
        return new_data - ((c1.sum(axis=0) + c2.sum(axis=0))/(c1.shape[0] + c2.shape[0]))
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'PHCClassifier':
        """Fit the PHC classifier"""
        # Clean and prepare texts
        cleaned_texts = [self._clean_text(text) for text in X]
        
        # Get vocabulary if not provided
        if self.vocabulary is None:
            self.vocabulary = self._get_vocab(' '.join(cleaned_texts))
        
        # Split data by class
        text1 = [t for t, l in zip(cleaned_texts, y) if l == 'human']
        text2 = [t for t, l in zip(cleaned_texts, y) if l == 'ChatGPT']
        
        # Convert to document-term matrices
        corpus1 = self._doc_to_dtm(text1, self.vocabulary)
        corpus2 = self._doc_to_dtm(text2, self.vocabulary)
        
        # Calculate statistics
        lam_1 = self._estimate_poisson(corpus1)
        lam_2 = self._estimate_poisson(corpus2)
        
        sx = np.std(corpus1, axis=0) + 1e-10
        sy = np.std(corpus2, axis=0) + 1e-10
        
        z = (lam_1 - lam_2) / np.sqrt((sx**2/corpus1.shape[0]) + (sy**2/corpus2.shape[0]))
        sign_z = np.sign(z)
        
        if self.ef_corr:
            z_n = (z - np.mean(z)) / np.std(z)
        else:
            z_n = z
            
        # Calculate p-values and HC
        pvals = 1 - norm.cdf(np.abs(z_n))
        _, i_star = self._higher_criticism(pvals)
        
        # Store selected words info
        selected_indices = pvals <= np.sort(pvals)[i_star]
        self.selected_words_info = []
        
        for idx, is_selected in enumerate(selected_indices):
            if is_selected:
                self.selected_words_info.append({
                    'word': self.vocabulary[idx],
                    'effect_size': abs(z[idx]),
                    'p_value': pvals[idx],
                    'direction': 'human' if z[idx] > 0 else 'ChatGPT',
                    'sign': sign_z[idx]
                })
        # Sort by effect size
        self.selected_words_info.sort(key=lambda x: x['effect_size'], reverse=True)
        print(f"Using {len(self.selected_words_info)} words!")
        for word_info in self.selected_words_info[:20]:  # Show top 20 words
            print(f"{word_info['word']:<20} {word_info['effect_size']:>12.4f} {word_info['p_value']:>10.4f} {word_info['direction']:>15}")
        return self
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Predict using the PHC classifier"""
        if not self.selected_words_info:
            raise ValueError("Classifier must be fitted before making predictions")
        
        # Clean and transform test texts
        cleaned_texts = [self._clean_text(text) for text in X]
        test_matrix = self._doc_to_dtm(cleaned_texts, self.vocabulary)
        
        # Get selected features and their signs
        selected_indices = [i for i, word in enumerate(self.vocabulary) 
                          if any(w['word'] == word for w in self.selected_words_info)]
        signs = np.array([next(w['sign'] for w in self.selected_words_info 
                         if w['word'] == self.vocabulary[i]) 
                         for i in selected_indices])
        
        # Make predictions
        test_selected = test_matrix[:, selected_indices]
        scores = signs * test_selected
        predictions = np.where(scores.sum(axis=1) > 0, 'human', 'ChatGPT')
        
        return predictions
    
    def get_discriminative_words(self) -> List[Dict]:
        """Return discriminative words and their statistics"""
        return self.selected_words_info

class PHCLLMClassifier:
    def __init__(self, model_name: str = "gpt-4o-mini", method: str = "llm_only", base_url = 'https://api.deepseek.com'):
        """
        Initialize classifier with specified method
        method options: "llm_only" or "llm_phc"
        """
        self.model_name = model_name
        self.method = method
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],base_url  = base_url)
        # Initialize PHC only if using combined method
        self.phc = None
        if self.method == "llm_phc":
            self.phc = PHCClassifier()
        
    def _generate_prompt(self, text: str) -> str:
        """Generate prompt based on method"""
        if self.method == "llm_phc" and self.phc and self.phc.selected_words_info:
            # Get discriminative words from PHC
            human_words = [w['word'] for w in self.phc.selected_words_info if w['direction'] == 'human']
            ai_words = [w['word'] for w in self.phc.selected_words_info if w['direction'] == 'ChatGPT']
            discriminative_info = (
            f"\n\nBased on statistical analysis, here are characteristic words:"
            f"\nTypically human-written: {', '.join(human_words)}"
            f"\nTypically AI-written: {', '.join(ai_words)}"
        )
            system_prompt = f"""You are a classifier that determines whether text is human-written or written by a human but AI-edited. 
    You will be given academic abstracts that are either written by humans or edited by AI.{discriminative_info}
    
    Respond with exactly one word: either 'human' for human-written text or 'ChatGPT' for AI-edited text. 
    Be as accurate as possible in your classification, considering both the writing style and the presence of characteristic words."""
    
            
#             prompt = f"""Analyze if this text is written by a human or AI.
# Words more common in human text: {', '.join(human_words)}
# Words more common in AI text: {', '.join(ai_words)}


        else:
            system_prompt =   """You are a classifier that determines whether text is human-written or AI-edited. 
            Respond with exactly one word: either 'human' for human-written text or 'ChatGPT' for AI-edited text. 
            Be as accurate as possible in your classification."""

        return system_prompt
        
    def fit(self, X: pd.Series, y: pd.Series) -> 'PHCLLMClassifier':
        """Train PHC if using combined method"""
        if self.method == "llm_phc" and self.phc:
            self.phc.fit(X, y)
        return self
        
    def predict(self, X: pd.Series) -> np.ndarray:
        """Make predictions using appropriate method"""
        predictions = []
        
        for i, text in enumerate(X):
            if i % 10 == 0:  # Progress update every 10 documents
                print(f"Processing document {i+1}/{len(X)}")
            
            try:
                # Try LLM prediction
                system_prompt = self._generate_prompt(text)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this text: {text}"}
                ]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1,
                    stream=False,
                    temperature=0
                )
                pred = response.choices[0].message.content.strip().lower()
                if pred not in ['human', 'chatgpt']:
                    print(f"Warning: Unexpected prediction '{pred}', defaulting to 'ChatGPT'")
                    pred = 'ChatGPT'
                
                predictions.append('human' if 'human' in pred else 'ChatGPT')
                #time.sleep(1)  # Rate limiting
                
            except Exception as e:
                if self.method == "llm_phc" and self.phc:
                    # Use PHC as fallback if available
                    print(f"LLM failed: {str(e)}. Using PHC fallback.")
                    phc_pred = self.phc.predict(pd.Series([text]))
                    predictions.append(phc_pred[0])
                else:
                    print(f"LLM failed: {str(e)}. Using default prediction.")
                    predictions.append('human')
                    
        return np.array(predictions)
class DetectGPTClassifier:
    """Wrapper for GPT2PPLV2 to match the classifier interface"""
    def __init__(self, model_name="gpt2-medium"):
        self.model = GPT2PPLV2(
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_id=model_name
        )
    
    def fit(self, X_train: pd.Series, y_train: pd.Series):
        """DetectGPT doesn't require training"""
        return self
    
    def predict(self, X_test: pd.Series) -> np.ndarray:
        """Predict using DetectGPT"""
        predictions = []
        for text in X_test:
            result, _ = self.model(text, 100, "v1.1")  # chunk_value=100, version="v1.1"
            predictions.append(result['label'])
        return np.array(predictions)

class TopicScoreLLMClassifier:
    """Combined Topic Score and LLM classifier implementation"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", n_topics: int = 3, 
                 threshold: float = 0.0001, method: str = "llm_ts"):
        """
        Initialize classifier with specified parameters
        
        Args:
            model_name: Name of the LLM model to use
            n_topics: Number of topics for Topic Score
            threshold: Threshold for word filtering
            method: Classification method ('llm_ts' or 'llm_only')
        """
        self.model_name = model_name
        self.method = method
        self.client = OpenAI()
        
        # Initialize Topic Score
        self.topic_score = TopicScoreClassifier(
            n_topics=n_topics,
            threshold=threshold
        )
        
        # Store discriminative words
        self.human_words = None
        self.chatgpt_words = None
        
    def _filter_words(self, A_h, A_ai, vocabulary=None):
        """Filter words and determine their association with human/AI text"""
        # Get vocabulary from vectorizer
        feature_names = self.topic_score.vectorizer.get_feature_names_out()
        
        # Compute L1 norm of differences for each word
        differences = np.linalg.norm(A_h - A_ai, ord=1, axis=1)
        
        # Find indices where difference exceeds threshold
        selected_indices = np.where(differences > self.topic_score.threshold)[0]
        
        # Calculate normalized sums for direction determination
        h_sums = np.sum(A_h[selected_indices, :], axis=1)
        ai_sums = np.sum(A_ai[selected_indices, :], axis=1)
        
        # Filter out numbers and short words
        valid_indices = []
        feature_list = []
        for idx in selected_indices:
            word = feature_names[idx]
            if len(word) > 2 and not word.isdigit():
                valid_indices.append(idx)
                feature_list.append(word)
                
        selected_indices = np.array(valid_indices)
        
        # Recalculate sums for valid indices only
        h_sums = np.sum(A_h[selected_indices, :], axis=1)
        ai_sums = np.sum(A_ai[selected_indices, :], axis=1)
        
        # Determine directions based on sums
        directions = []
        human_words = []
        chatgpt_words = []
        
        for i, (h_sum, ai_sum) in enumerate(zip(h_sums, ai_sums)):
            word = feature_list[i]
            score = differences[selected_indices[i]]
            
            if h_sum > ai_sum:
                directions.append('human')
                human_words.append((word, score))
            else:
                directions.append('ChatGPT')
                chatgpt_words.append((word, score))
        
        #print("human_words = ", human_words[:10])
        return {
            'human_words': human_words,
            'chatgpt_words': chatgpt_words,
            'selected_indices': selected_indices,
            'differences': differences[selected_indices]
        }
    
    def _generate_prompt(self, text: str) -> str:
        """Generate prompt incorporating Topic Score insights"""
        if self.method == "llm_ts" and self.human_words and self.chatgpt_words:
            # Get top words sorted by effect size
            #print("human = ",self.human_words)
            human_word_list = sorted(self.human_words, 
                                   key=lambda x: x[1], 
                                   reverse=True)[:20]
            ai_word_list = sorted(self.chatgpt_words, 
                                key=lambda x: x[1], 
                                reverse=True)[:20]
            
            human_words = ', '.join([w[0] for w in human_word_list])
            ai_words = ', '.join([w[0] for w in ai_word_list])
            #print("human = ",human_words)
            
            discriminative_info = (
                f"\n\nBased on topic analysis, here are characteristic words:"
                f"\nTypically human-written: {human_words}"
                f"\nTypically AI-written: {ai_words}"
            )
        else:
            discriminative_info = ""
            
        system_prompt = f"""You are a classifier that determines whether text is human-written or AI-edited.
    You will be given academic abstracts that are either written by humans or edited by AI.{discriminative_info}
    Respond with exactly one word: either 'human' for human-written text or 'ChatGPT' for AI-edited text. 
    Be as accurate as possible in your classification, considering both the writing style and the presence of characteristic words."""
        
        return system_prompt
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'TopicScoreLLMClassifier':
        """Train the classifier"""
        # Train Topic Score component
        self.topic_score.fit(X, y)
        
        # Extract discriminative words using modified filter
        if hasattr(self.topic_score, 'A_h') and hasattr(self.topic_score, 'A_ai'):
            filtered_results = self._filter_words(
                self.topic_score.A_h,
                self.topic_score.A_ai
            )
            
            # Sort words by score and get top ones
            self.human_words = sorted(filtered_results['human_words'], 
                                    key=lambda x: x[1], 
                                    reverse=True)
            self.chatgpt_words = sorted(filtered_results['chatgpt_words'], 
                                      key=lambda x: x[1], 
                                      reverse=True)
            
            # Print top discriminative words
            # print("\nTop discriminative words:")
            # print("\nHuman words:")
            for word, score in sorted(self.human_words, key=lambda x: x[1], reverse=True)[:20]:
                print(f"HUMAN WORDS TS: {word:<20} {score:>10.4f}")
            print("\nChatGPT words:")
            for word, score in sorted(self.chatgpt_words, key=lambda x: x[1], reverse=True)[:20]:
                print(f"GPT WORDS: {word:<20} {score:>10.4f}")
        
        return self
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Make predictions using the combined classifier"""
        predictions = []
        
        for i, text in enumerate(X):
            if i % 10 == 0:
                print(f"Processing document {i+1}/{len(X)}")
            
            try:
                # Generate prompt and get LLM prediction
                system_prompt = self._generate_prompt(text)
                #print("prompt = ", system_prompt)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this text: {text}"}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1,
                    temperature=0
                )
                
                pred = response.choices[0].message.content.strip().lower()
                if pred not in ['human', 'chatgpt']:
                    print(f"Warning: Unexpected prediction '{pred}', defaulting to 'ChatGPT'")
                    pred = 'ChatGPT'
                
                predictions.append(pred)
                
            except Exception as e:
                # Fallback to Topic Score prediction
                print(f"LLM failed: {str(e)}. Using Topic Score fallback.")
                ts_pred = self.topic_score.predict(pd.Series([text]))
                predictions.append(ts_pred[0])
        
        return np.array(predictions)

def run_comparison(data_loader: TextDataLoader,
                  methods: Dict[str, TextClassifier],
                  cv_folds: int = 5,
                  test_size: float = 0.2,
                  use_cv=False) -> Dict:
    """Run comparison between multiple classification methods"""
    results = {}
    data, labels = data_loader.load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'],
        test_size=test_size,
        stratify=data['label']
    )

    # Run each method
    for method_name, classifier in methods.items():
        print(f"\nRunning {method_name}...")
        try:
            if method_name == 'TopicScore' and use_cv:
                # Define parameter combinations to try
                n_topics_values = [3, 4, 5]
                threshold_values = [0.001, 0.01]
                
                best_accuracy = 0
                best_params = None
                
                # Try each parameter combination with cross validation
                for n_topics in n_topics_values:
                    for threshold in threshold_values:
                        print(f"\nTrying n_topics={n_topics}, threshold={threshold}")
                        
                        cv_accuracies = []
                        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        
                        # Perform cross validation for this parameter combination
                        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                            # Split data for this fold
                            X_train_fold = X_train.iloc[train_idx]
                            y_train_fold = y_train.iloc[train_idx]
                            X_val_fold = X_train.iloc[val_idx]
                            y_val_fold = y_train.iloc[val_idx]
                            
                            # Create and train classifier with current parameters
                            fold_classifier = TopicScoreClassifier(n_topics=n_topics, threshold=threshold)
                            fold_classifier.fit(X_train_fold, y_train_fold)
                            
                            # Evaluate on validation fold
                            val_predictions = fold_classifier.predict(X_val_fold)
                            fold_accuracy = accuracy_score(y_val_fold, val_predictions)
                            cv_accuracies.append(fold_accuracy)
                        
                        # Calculate mean CV accuracy for this parameter combination
                        mean_cv_accuracy = np.mean(cv_accuracies)
                        print(f"Mean CV accuracy: {mean_cv_accuracy:.4f}")
                        
                        # Update best parameters if this combination is better
                        if mean_cv_accuracy > best_accuracy:
                            best_accuracy = mean_cv_accuracy
                            best_params = {'n_topics': n_topics, 'threshold': threshold}
                
                print(f"\nBest parameters found: {best_params}")
                print(f"Best CV accuracy: {best_accuracy:.4f}")
                
                # Train final model with best parameters on full training set
                best_classifier = TopicScoreClassifier(**best_params)
                best_classifier.fit(X_train, y_train)
                predictions = best_classifier.predict(X_test)
                
            else:
                # Original training process for other methods
                classifier.fit(X_train, y_train)
                predictions = classifier.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)
            
            results[method_name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'predictions': predictions
            }
            
            # Add best parameters for TopicScore
            if method_name == 'TopicScore' and use_cv:
                results[method_name]['best_parameters'] = best_params
                results[method_name]['best_cv_accuracy'] = best_accuracy
            
            # Print results
            print(f"\nFinal {method_name} Test Results:")
            print(f"Test Accuracy: {accuracy:.4f}")
            print("Confusion Matrix:")
            print(cm)
            
        except Exception as e:
            print(f"Error in {method_name}: {str(e)}")
            continue
            
    return results


if __name__ == "__main__":
    # Example usage with all methods
    parser = argparse.ArgumentParser(description='Text Classification Framework')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--data-format', choices=['coauthor', 'csv'], required=True)
    parser.add_argument('--llm-model', default='gpt-4o-mini')
    args = parser.parse_args()
    
    # Initialize data loader
    if args.data_format == 'coauthor':
        data_loader = CoAuthorDataLoader(args.data_path)
    else:
        data_loader = StandardCSVDataLoader(args.data_path)
    
    # Initialize all classifiers
    methods = {
        #'TopicScore': TopicScoreClassifier(n_topics=3, threshold=0.0001),
        #'TF-IDF': TFIDFClassifier()
        'PHC': PHCClassifier(),
        #'LLM': PHCLLMClassifier(model_name=args.llm_model, method="llm_only")
        # 'LLM+PHC': PHCLLMClassifier(model_name=args.llm_model, method="llm_phc"),
        'TopicScoreLLM': TopicScoreLLMClassifier(
        model_name=args.llm_model,
        n_topics=3,
        threshold=0.0001,
        method="llm_ts")
    }
    # methods = {
    #     'TopicScore': TopicScoreClassifier(n_topics=4, threshold=0.001),
    #     'PHC': PHCClassifier(),
    #     'DetectGPT': DetectGPTClassifier(
    #     model_name="gpt2-medium"
    # )
    # }

    
    # Run comparison
    results = run_comparison(data_loader, methods)
    
    # Print results
    print("\nResults Summary:")
    print("="*50)
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        if 'confusion_matrix' in metrics:
            print("Confusion Matrix:")
            print(metrics['confusion_matrix'])
