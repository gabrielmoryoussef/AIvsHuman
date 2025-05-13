import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
from typing import List, Tuple, Dict, Union, Optional
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from gpt2model import GPT2PPLV2
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
from anthropic import Anthropic
from typing import List, Optional
import random, re, math
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers import (GPT2LMHeadModel, GPT2TokenizerFast,
                          T5ForConditionalGeneration, T5Tokenizer)


from pydetectgpt import detect_ai_text

# Ignore transformer warnings
warnings.filterwarnings("ignore")

_TAG = re.compile(r'prediction\s*:\s*(human|ChatGPT)', re.I)

def extract_final_label(text: str, default: str = "ChatGPT") -> str:
    """
    Return the label ('human' or 'ChatGPT') that appears
    in the **last** 'prediction:<label>' tag.
    """
    hits = _TAG.findall(text)
    if hits:
        return hits[-1]         # keep the final occurrence
    print("Warning: no prediction tag found; defaulting.")
    return default

def tolist_safe(x):
    return x.tolist() if hasattr(x, "tolist") else x

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
    def __init__(self, n_topics: int = 3, share_weights=False, threshold: float = 0.0001, **kwargs):
        super().__init__(**kwargs)
        self.n_topics = n_topics
        self.threshold = threshold
        self.A_h = None
        self.A_ai = None
        self.selected_indices = None
        self.vocabulary = None
        self.share_weights = share_weights
        self.human_words = None
        self.chatgpt_words = None
        
    def _run_topic_score(self, K, X, K0=None, m=None, Mquantile=0, num_restart=1, seed=None):
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
        feature_names = self.vectorizer.get_feature_names_out()
        differences = np.linalg.norm(A_h - A_ai, ord=1, axis=1)
        selected_indices = np.where(differences > self.threshold)[0]

        h_sums = np.sum(A_h[selected_indices, :], axis=1)
        ai_sums = np.sum(A_ai[selected_indices, :], axis=1)

        valid_indices = []
        feature_list = []
        for idx in selected_indices:
            word = feature_names[idx]
            if len(word) > 2 and not word.isdigit():
                valid_indices.append(idx)
                feature_list.append(word)

        selected_indices = np.array(valid_indices)
        h_sums = np.sum(A_h[selected_indices, :], axis=1)
        ai_sums = np.sum(A_ai[selected_indices, :], axis=1)

        human_words = []
        chatgpt_words = []

        for i, (h_sum, ai_sum) in enumerate(zip(h_sums, ai_sums)):
            word = feature_list[i]
            score = differences[selected_indices[i]]
            if h_sum > ai_sum:
                human_words.append((word, score))
            else:
                chatgpt_words.append((word, score))

        return {
            'human_words': human_words,
            'chatgpt_words': chatgpt_words,
            'selected_indices': selected_indices,
            'differences': differences[selected_indices]
        }
        
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
    
    def fit(self, X_train: pd.Series, y_train: List[str]):
        X = self.vectorizer.fit_transform(X_train)
        self.vocabulary = self.vectorizer.get_feature_names_out()

        mask_human = np.array(y_train) == 'human'
        X_human = X[mask_human].toarray().T
        X_ai = X[~mask_human].toarray().T

        X_human = X_human / (X_human.sum(axis=0) + 1e-10)
        X_ai = X_ai / (X_ai.sum(axis=0) + 1e-10)

        if self.share_weights:
            D_combined = np.vstack([X_human, X_ai]) / 2
            results_combined = self._run_topic_score(self.n_topics, D_combined)
            p = X_human.shape[0]
            A_combined = results_combined['A_hat']
            self.A_h = A_combined[:p, :]
            self.A_ai = A_combined[p:, :]
        else:
            results_h = self._run_topic_score(self.n_topics, X_human)
            results_ai = self._run_topic_score(self.n_topics, X_ai)
            self.A_h = results_h['A_hat']
            self.A_ai = results_ai['A_hat']

        filtered_results = self._filter_words(self.A_h, self.A_ai, self.vocabulary)
        self.selected_indices = filtered_results['selected_indices']
        self.human_words = filtered_results['human_words']
        self.chatgpt_words = filtered_results['chatgpt_words']

        return self
        
    def predict(self, X_test: pd.Series) -> List[str]:
        X = self.vectorizer.transform(X_test)
        predictions = []
        for doc in X:
            doc_vector = doc.toarray()[0]
            if np.sum(doc_vector) != 0:
                doc_vector = doc_vector / np.sum(doc_vector)
            d_S = doc_vector[self.selected_indices]

            error_human, _ = self._solve_reconstruction(self.A_h[self.selected_indices], d_S)
            error_ai, _ = self._solve_reconstruction(self.A_ai[self.selected_indices], d_S)

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
    """Paired‑Human Classifier with Higher Criticism and robust vocabulary filtering."""

    # ------------------------------------------------------------------
    def __init__(self,
                 max_vocab_length: int = 1_000,
                 gamma: float = 0.25,
                 threshold: float = 0.99,
                 ef_corr: bool = True,
                 min_doc_freq: int = 3,
                 min_tfidf: float = 1e-4,
                 use_tfidf: bool = False):          # <-- new flag
        # user hyper-params
        self.max_vocab_length = max_vocab_length
        self.gamma = gamma
        self.threshold = threshold
        self.ef_corr = ef_corr
        self.min_doc_freq = max(1, min_doc_freq)
        self.min_tfidf = max(0.0, min_tfidf)
        self.use_tfidf = use_tfidf              # <-- store flag
        # learned
        self.vocabulary: Optional[List[str]] = None
        self.selected_words_info: List[Dict] = []
        self.vectorizer: Optional[CountVectorizer] = None
    # ------------------------------------------------------------------
    def set_vocabulary(self, vocab: List[str]):
        """Force a fixed vocabulary (e.g. built on full corpus)."""
        self.vocabulary = vocab[: self.max_vocab_length]

    # ------------------------- text utilities -------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub("All rights", "", text)
        text = re.sub("reserved", "", text)
        text = re.sub("copyright", "", text)
        text = re.sub("[^A-Za-z ]", "", text)
        return text.lower()

    @staticmethod
    def _topk_frequent(tokens: List[str], k: int) -> List[str]:
        cnt = Counter(tokens)
        heap = [[-f, w] for w, f in cnt.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(min(k, len(heap)))]

    def _doc_to_dtm(self, texts: List[str], vocab: List[str]) -> np.ndarray:
        stops = list(text_.ENGLISH_STOP_WORDS) + ["Abstract"]
        self.vectorizer = CountVectorizer(tokenizer=lambda s: s.split(),
                                          vocabulary=vocab,
                                          stop_words=stops)
        return self.vectorizer.transform(texts).toarray()

    @staticmethod
    def _estimate_poisson(corpus: np.ndarray) -> np.ndarray:
        return corpus.mean(axis=0)

    def _higher_criticism(self, pvals: np.ndarray) -> Tuple[float, int]:
        pvals = np.sort(pvals[pvals <= self.threshold])
        N = len(pvals)
        best, idx = -1_000.0, 0
        for i in range(1, int(self.gamma * N) + 1):
            if pvals[i-1] >= 1/N:
                num = np.sqrt(N) * ((i/N) - pvals[i-1])
                den = np.sqrt((i/N) * (1 - i/N))
                val = num / den
                if val > best:
                    best, idx = val, i
        return best, idx

    # ---------------------- vocabulary builder -----------------------
    def _build_vocabulary(self, cleaned: List[str]):
        base_stops = list(text_.ENGLISH_STOP_WORDS) + ["Abstract"]
        cv = CountVectorizer(tokenizer=lambda s: s.split(), stop_words=base_stops)
        dtm = cv.fit_transform(cleaned)
        vocab_arr = np.array(cv.get_feature_names_out())

        # document-frequency filter
        df = (dtm > 0).sum(axis=0).A1
        keep_df = df >= self.min_doc_freq

        # optional TF-IDF filter
        if self.use_tfidf and self.min_tfidf > 0:
            tfidf = TfidfVectorizer(
                vocabulary=vocab_arr,
                tokenizer=lambda s: s.split(),
                stop_words=base_stops,
            )
            tfidf_mat = tfidf.fit_transform(cleaned)
            mean_tfidf = np.asarray(tfidf_mat.mean(axis=0)).ravel()
            keep_tf = mean_tfidf >= self.min_tfidf
        else:
            keep_tf = np.ones_like(keep_df, dtype=bool)   # no TF-IDF filter

        keep = keep_df & keep_tf
        vocab_filtered = vocab_arr[keep]
        if len(vocab_filtered) < 50:                      # fall-back
            vocab_filtered = vocab_arr

        freq_order = np.argsort(-dtm[:, keep].sum(axis=0).A1)
        self.vocabulary = vocab_filtered[freq_order][: self.max_vocab_length].tolist()
    # ------------------------------------------------------------------
    def fit(self, X: pd.Series, y: pd.Series, *, external_vocab: Optional[List[str]] = None):
        cleaned = [self._clean_text(t) for t in X]

        # 1) vocabulary
        if external_vocab is not None:
            self.vocabulary = external_vocab[: self.max_vocab_length]
        elif self.vocabulary is None:
            self._build_vocabulary(cleaned)

        # 2) split docs
        txt_h = [t for t, lbl in zip(cleaned, y) if lbl == "human"]
        txt_ai = [t for t, lbl in zip(cleaned, y) if lbl == "ChatGPT"]

        corp_h = self._doc_to_dtm(txt_h, self.vocabulary)
        corp_ai = self._doc_to_dtm(txt_ai, self.vocabulary)

        # 3) z‑scores
        lam_h, lam_ai = self._estimate_poisson(corp_h), self._estimate_poisson(corp_ai)
        sx = corp_h.std(axis=0) + 1e-10
        sy = corp_ai.std(axis=0) + 1e-10
        z = (lam_h - lam_ai) / np.sqrt((sx**2 / corp_h.shape[0]) + (sy**2 / corp_ai.shape[0]))
        sign_z = np.sign(z)
        z_n = (z - z.mean()) / z.std() if self.ef_corr else z
        pvals = 1 - norm.cdf(np.abs(z_n))
        _, i_star = self._higher_criticism(pvals)

        sel_mask = pvals <= np.sort(pvals)[i_star]
        self.selected_words_info = [
            {
                "word": self.vocabulary[i],
                "effect_size": abs(z[i]),
                "p_value": float(pvals[i]),
                "direction": "human" if z[i] > 0 else "ChatGPT",
                "sign": sign_z[i],
            }
            for i in np.where(sel_mask)[0]
        ]
        self.selected_words_info.sort(key=lambda d: d["effect_size"], reverse=True)
        #print(self.selected_words_info)
        print(f"Using {len(self.selected_words_info)} discriminative words (|V|={len(self.vocabulary)})")
        print("\nTop discriminative words:")
        print(self.selected_words_info[:20])
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.Series, *, return_scores: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not self.selected_words_info:
            raise RuntimeError("PHCClassifier must be fitted first.")

        cleaned = [self._clean_text(t) for t in X]
        dtm = self._doc_to_dtm(cleaned, self.vocabulary)
        sel_idx = [i for i, w in enumerate(self.vocabulary)
                    if any(info["word"] == w for info in self.selected_words_info)]
        signs = np.array([next(info["sign"] for info in self.selected_words_info if info["word"] == self.vocabulary[i])
                          for i in sel_idx])
        scores = dtm[:, sel_idx] @ signs  # shape (n,)
        preds = np.where(scores > 0, "human", "ChatGPT")
        return (preds, scores) if return_scores else preds

    # ------------------------------------------------------------------
    def get_discriminative_words(self) -> List[Dict]:
        return self.selected_words_info



class PHCLLMClassifier:
    def __init__(self, model_name: str = "gpt-4o-mini", method: str = "llm_only", base_url = 'https://api.deepseek.com'):
        """
        Initialize classifier with specified method
        method options: "llm_only" or "llm_phc"
        """
        self.model_name = model_name
        self.method = method
        if 'llama' in self.model_name:
        # Initialize NVIDIA clients with multiple API keys
            api_key = os.getenv("NVIDIA_API_KEY_1")
       
        # Create client instances
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key
            )
        elif 'claude' in self.model_name:
            
            api_key = os.getenv('CLAUDE_API_KEY')
            if api_key is None:
                raise ValueError("No API key provided and ANTHROPIC_API_KEY not found in environment")
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],base_url  = base_url)
        # Initialize PHC only if using combined method
        self.phc = None
        if self.method == "llm_phc":
            self.phc = PHCClassifier()
            
    def _generate_prompt(self, text: str) -> str:
        """Generate the best-scoring prompt using PHC prediction + words + deferral logic."""
        if self.method != "llm_phc" or self.phc is None or self.phc.selected_words_info is None:
            # Default fallback prompt
            return (
                "You are a classifier that determines whether a document is human-written or edited by ChatGPT. "
                "Respond with exactly one word: either 'human' or 'ChatGPT'."
            )

        # Get PHC prediction for this specific text
        try:
            phc_prediction = self.phc.predict(pd.Series([text]))[0]
        except Exception as e:
            print(f"PHC prediction failed: {e}")
            phc_prediction = "unknown"

        # Extract top PHC words by direction
        
        
        human_words = [w['word'] for w in self.phc.selected_words_info if w['direction'] == 'human'][:30]
        ai_words = [w['word'] for w in self.phc.selected_words_info if w['direction'] == 'ChatGPT'][:30]
        
        # prompt = f"""You are an expert forensic linguist.

        # Task ─────────────────────────────────────────────
        # Decide whether the following document was
        # • written entirely by a human      → label “human”
        # • edited or rewritten by ChatGPT   → label “ChatGPT”

        # Prior knowledge ────────────────────────────────
        # A trusted statistical detector (PHC) has already analysed the text:

        # • PHC’s prediction:  {phc_prediction}

        # Discriminative vocabulary discovered by PHC:
        # • Common in human documents   :  {', '.join(human_words)}
        # • Common in ChatGPT documents :  {', '.join(ai_words)}

        # Guidelines for you ─────────────────────────────
        # 1. Start from PHC’s prediction.  
        # Override it ONLY if you see contrary evidence .
        # 2. Give ONE token answer: exactly “human” or “ChatGPT”.  
        # Do NOT include sentences, punctuation, or other text.

        # Document ───────────────────────────────────────
        # {text}
        # ───────────────────────────────────────────────
        # Answer with one word now:"""

#         prompt = f"""You are an expert forensic linguist.

#         Task ─────────────────────────────────────────────
#         Decide whether the following document was
#         • written entirely by a human      → label “human”
#         • Written by a human but then edited or rewritten by ChatGPT   → label “ChatGPT”

#         Prior knowledge ────────────────────────────────
#         A trusted statistical detector (PHC) has already analysed the text:


#         Discriminative vocabulary discovered by PHC:
#         • Common in human documents   :  {', '.join(human_words)}
#         • Common in documents edited by ChatGPT:  {', '.join(ai_words)}

        
#         Think step-by-step.

# When you are done, output **exactly one final line**:

# prediction:<label>

# where <label> is either human or ChatGPT (case-sensitive).
#         Document ───────────────────────────────────────
#         {text}
#         ───────────────────────────────────────────────
#         """
        prompt = f"""You are an expert forensic linguist.

        Task ─────────────────────────────────────────────
        Decide whether the following document was
        • written entirely by a human      → label “human”
        • Written by a human but then edited or rewritten by ChatGPT   → label “ChatGPT”

        Prior knowledge ────────────────────────────────
        A trusted statistical detector (PHC) has already analysed the text:


        Discriminative vocabulary discovered by PHC:
        • Common in human documents   :  {', '.join(human_words)}
        • Common in documents edited by ChatGPT:  {', '.join(ai_words)}

        
        

Answer with exatcly the following format without outputing your reasoning.

prediction:<label>

where <label> is either human or ChatGPT (case-sensitive).
        Document ───────────────────────────────────────
        {text}
        ───────────────────────────────────────────────
        """
        

        return prompt



        
#     def _generate_prompt(self, text: str) -> str:
#         """Generate prompt based on method"""
#         if self.method == "llm_phc" and self.phc and self.phc.selected_words_info:
#             # Get discriminative words from PHC
#             human_words = [w['word'] for w in self.phc.selected_words_info if w['direction'] == 'human'][:20]
#             ai_words = [w['word'] for w in self.phc.selected_words_info if w['direction'] == 'ChatGPT'][:20]
            
#             discriminative_info = (
#             f"\n\nBased on statistical analysis, here are characteristic words:"
#             f"\nTypically human-written: {', '.join(human_words)}"
#             f"\nTypically corresponding to ChatGPT edited: {', '.join(ai_words)}"
#         )
#     #         system_prompt = f"""You are a classifier that determines whether text is human-written or written by a human but AI-edited. 
#     # You will be given academic abstracts that are either written by humans or edited by AI.{discriminative_info}
    
#     # Respond with exactly one word: either 'human' for human-written text or 'ChatGPT' for AI-edited text. 
#     # Be as accurate as possible in your classification, considering both the writing style and the presence of characteristic words."""
#             system_prompt = f"""You are a classifier that determines whether a text is human-written or AI-edited.
# You will be given  documents that are either written by humans or that were edited by AI. After running the Higher Criticism on the data \
#     it gave the following discriminative info. {discriminative_info}
# Respond with exactly one word: either 'human' if you think the following text is human-written text or 'ChatGPT' if you think it is AI-edited. 
# Be as accurate as possible in your classification, considering both the writing style and especially the presence of the discriminative words."""
        
    
            
# #             prompt = f"""Analyze if this text is written by a human or AI.
# # Words more common in human text: {', '.join(human_words)}
# # Words more common in AI text: {', '.join(ai_words)}


#         else:
#             # system_prompt =   """You are a classifier that determines whether text is human-written or AI-edited. 
#             # Respond with exactly one word: either 'human' for human-written text or 'ChatGPT' for AI-edited text. 
#             # Be as accurate as possible in your classification."""
#             system_prompt =   """You are a classifier that determines whether the given movie review is human-written or it was edited by ChatGPT. 
#             Respond with exactly one word: either 'human' for human-written text or 'ChatGPT' for AI-edited text. 
#             Be as accurate as possible in your classification."""

#         return system_prompt
        
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
                # Get system prompt
                system_prompt = self._generate_prompt(text)

                if 'claude' in self.model_name:
                    # For Claude, system message is a top-level parameter
                    response = self.client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        system=system_prompt,  # System message as top-level parameter
                        messages=[
                            {"role": "user", "content": f"Classify this text: {text}"}
                        ],
                        max_tokens=10,
                        temperature=0.1
                    )
                    
                    pred = response.content[0].text.strip().lower()
                    #print("pred = ",pred)
                    label = extract_final_label(pred)
                    print(label)
                else:
                    # For other models like OpenAI, system is a role
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Classify this text: {text}"}
                        ],
                        max_tokens=10,
                        stream=False,
                        temperature=0
                    )
                    pred = response.choices[0].message.content.strip().lower()
                    #print("pred = ",pred)
                    label = extract_final_label(pred)
                    print(label)
                    
                # if pred not in ['human', 'chatgpt']:
                #     print(f"Warning: Unexpected prediction '{pred}', defaulting to 'ChatGPT'")
                #     pred = 'ChatGPT'
                    
                predictions.append('human' if 'human' in label else 'ChatGPT')
                
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

class TopicScoreLLMClassifier:
    def __init__(self, model_name: str = "gpt-4o-mini", n_topics: int = 3, 
                 threshold: float = 0.01, method: str = "llm_ts"):
        self.model_name = model_name
        self.method = method
        if 'llama' in self.model_name:
        # Initialize NVIDIA clients with multiple API keys
            api_key = os.getenv("NVIDIA_API_KEY_1")
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key
            )
        else:
            self.client = OpenAI()
        self.topic_score = TopicScoreClassifier(n_topics=n_topics, threshold=threshold)
        self.human_words = None
        self.chatgpt_words = None

    def _generate_prompt(self, text: str) -> str:
        if self.method == "llm_ts" and self.human_words and self.chatgpt_words:
            human_word_list = sorted(self.human_words, key=lambda x: x[1], reverse=True)[:20]
            ai_word_list = sorted(self.chatgpt_words, key=lambda x: x[1], reverse=True)[:20]

            human_words = ', '.join([w[0] for w in human_word_list])
            ai_words = ', '.join([w[0] for w in ai_word_list])

            discriminative_info = (
                f"\n\nBased on topic analysis, here are characteristic words:"
                f"\nTypically human-written: {human_words}"
                f"\nTypically AI-written: {ai_words}"
            )
        else:
            discriminative_info = ""

#         system_prompt = f"""You are a classifier that determines whether a text is human-written or AI-edited.
# You will be given  documents that are either written by humans or that were edited by AI. After running the Higher Criticism on the data \
#     it gave the following discriminative info. {discriminative_info}
# Respond with exactly one word: either 'human' if you think the following text is human-written text or 'ChatGPT' if you think it is AI-edited. 
# Be as accurate as possible in your classification, considering both the writing style and especially the presence of the discriminative words."""
        system_prompt = f"""You are a classifier that determines whether a text is human-written or AI-edited.
You will be given  documents that are either written by humans or that were edited by AI. After running the Higher Criticism on the data \
    it gave the following discriminative info. {discriminative_info}
Classify the document below as either **human** (written entirely by a human) or **ChatGPT** (edited/rewritten by ChatGPT). 
"""
        
        return system_prompt

    def fit(self, X: pd.Series, y: pd.Series) -> 'TopicScoreLLMClassifier':
        self.topic_score.fit(X, y)
        self.human_words = self.topic_score.human_words
        self.chatgpt_words = self.topic_score.chatgpt_words
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        predictions = []
        for i, text in enumerate(X):
            if i % 10 == 0:
                print(f"Processing document {i+1}/{len(X)}")
            try:
                system_prompt = self._generate_prompt(text)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this text: {text}"}
                ]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1,
                    temperature=0.2
                )
                pred = response.choices[0].message.content.strip().lower()
                if pred not in ['human', 'chatgpt']:
                    print(f"Warning: Unexpected prediction '{pred}', defaulting to 'ChatGPT'")
                    pred = 'ChatGPT'
                predictions.append(pred)
            except Exception as e:
                print(f"LLM failed: {str(e)}. Using Topic Score fallback.")
                ts_pred = self.topic_score.predict(pd.Series([text]))
                predictions.append(ts_pred[0])
        return np.array(predictions)


class FastDetectGPT:

    def __init__(self):
        pass  # no model initialization needed
    def fit(self, X: List[str], y: List[str]):
        """No training necessary — placeholder to satisfy framework."""
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
      
        predictions = [
            "ChatGPT" if detect_ai_text(text) else "human"
            for text in texts
        ]
        return np.array(predictions)


class DetectGPTClassifier2:
    """
    DetectGPT-style detector with span-corruption perturbations produced by T5.

    Parameters
    ----------
    gpt_name   : LM used for scoring   (default 'gpt2-medium')
    t5_name    : LM used for noising   (default 't5-base')
    k          : #perturbations per text (default 10)
    mask_frac  : fraction of tokens to mask before refill (default 0.15)
    max_len    : max sequence length for GPT-2 chunks (default 1024)
    device     : 'cuda' / 'cpu'        (auto)
    """

    def __init__(self,
                 gpt_name: str = "gpt2",
                 t5_name : str = "t5-small",
                 k       : int = 1,
                 mask_frac: float = .15,
                 max_len : int = 1024,
                 device  : Optional[str] = None):

        self.device   = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using {self.device}")
        # scorer LM
        self.scorer   = GPT2LMHeadModel.from_pretrained(gpt_name).to(self.device)
        self.gtoker   = GPT2TokenizerFast.from_pretrained(gpt_name)
        self.gtoker.pad_token = self.gtoker.eos_token
        # perturbation LM
        self.noiser   = T5ForConditionalGeneration.from_pretrained(t5_name).to(self.device)
        self.ntoker   = T5Tokenizer.from_pretrained(t5_name)
        # hyper-params
        self.k         = k
        self.mask_frac = mask_frac
        self.max_len   = max_len
        self.threshold = None   # learned in fit()

    # ---------------------------------------------------------
    # helpers
    def _logprob(self, text: str) -> float:
        """Average token log-prob under GPT-2 (handles long docs)."""
        enc = self.gtoker(text, return_tensors="pt")
        ids = enc.input_ids[0]
        n   = ids.size(0)
        if n <= self.max_len:
            with torch.no_grad():
                out = self.scorer(ids.unsqueeze(0).to(self.device), labels=ids.unsqueeze(0).to(self.device))
            return -out.loss.item()            # negative NLL per token
        # long text → sliding window
        stride, ll_sum = self.max_len//2, 0.0
        for i in range(0, n, stride):
            beg = max(i+stride-self.max_len, 0); end = min(i+stride, n)
            trg_len = end - i
            inp  = ids[beg:end].unsqueeze(0).to(self.device)
            labs = inp.clone()
            labs[:, :-trg_len] = -100          # only predict last trg_len tokens
            with torch.no_grad():
                ll_sum -= self.scorer(inp, labels=labs).loss.item() * trg_len
            if end == n: break
        return ll_sum / n                      # average per token

    def _mask_spans(self, tokens: List[str]) -> str:
        """Randomly mask contiguous token spans à la span-corruption."""
        L = len(tokens); num_mask = max(1, int(self.mask_frac * L))
        mask_indices = sorted(random.sample(range(L), num_mask))
        masked = []; cur = 0; span_id = 0
        for i in range(L):
            if mask_indices and i == mask_indices[0]:
                masked.append(f"<extra_id_{span_id}>")
                span_id += 1
                while mask_indices and i == mask_indices[0]:
                    mask_indices.pop(0); i += 1
                if i >= L: break
            masked.append(tokens[i])
        masked.append(f"<extra_id_{span_id}>")
        return " ".join(masked)

    def _perturb(self, text: str) -> str:
        """Produce a single noisy paraphrase with T5 span infilling."""
        tok = self.ntoker
        tokens = tok.tokenize(text)[:512]                      # keep it short
        masked_input = self._mask_spans(tokens)
        input_ids = tok(masked_input, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            gen_ids = self.noiser.generate(input_ids, max_length=512, do_sample=True, top_p=.9, temperature=1.0)
        return tok.decode(gen_ids[0], skip_special_tokens=True)

    # ---------------------------------------------------------
    # API
    def fit(self, X_train: List[str], y_train: List[str]):
        """Find threshold that maximises accuracy on training data."""
        print(f"[DetectGPT] computing scores on {len(X_train)} training docs …")
        scores = []
        for txt in X_train:
            lp0 = self._logprob(txt)
            lps = [self._logprob(self._perturb(txt)) for _ in range(self.k)]
            scores.append(lp0 - float(np.mean(lps)))          # curvature score
        y_bin = np.array([0 if y.lower()=="human" else 1 for y in y_train])

        best_acc = 0.0
        for t in np.linspace(min(scores), max(scores), 100):
            preds = (np.array(scores) < t).astype(int)        # lower score ⇒ AI
            acc   = (preds == y_bin).mean()
            if acc > best_acc:
                best_acc, self.threshold = acc, t
        print(f"[DetectGPT]  chosen threshold {self.threshold:.4f}  (train acc {best_acc:.4f})")
        return self

    @torch.no_grad()
    def predict(self, X_test: List[str]) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Call fit() before predict().")
        preds = []
        for txt in X_test:
            lp0 = self._logprob(txt)
            lps = [self._logprob(self._perturb(txt)) for _ in range(self.k)]
            score = lp0 - float(np.mean(lps))
            preds.append("ChatGPT" if score < self.threshold else "human")
        return np.array(preds)






class DetectGPTClassifier:
    """Classifier based on GPT-2 perplexity, robust to long texts."""

    def __init__(self, model_name='gpt2-medium', device=None, threshold=None, max_length=1024):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.threshold = threshold  # Will be set during fit
        self.max_length = max_length

    def compute_perplexity(self, text):
        """Compute perplexity robustly by chunking long texts."""
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids[0]

        n_tokens = input_ids.size(0)
        if n_tokens <= self.max_length:
            input_ids = input_ids.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
            return torch.exp(loss).item()
        else:
            # Chunking for long sequences
            stride = self.max_length // 2
            nlls = []
            for i in range(0, n_tokens, stride):
                begin_loc = max(i + stride - self.max_length, 0)
                end_loc = min(i + stride, n_tokens)
                trg_len = end_loc - i  # Number of tokens to predict in this chunk

                input_ids_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(self.device)
                target_ids_chunk = input_ids_chunk.clone()
                target_ids_chunk[:, :-trg_len] = -100  # Mask tokens not to predict

                with torch.no_grad():
                    outputs = self.model(input_ids_chunk, labels=target_ids_chunk)
                    neg_log_likelihood = outputs.loss.item() * trg_len

                nlls.append(neg_log_likelihood)

                if end_loc == n_tokens:
                    break

            ppl = torch.exp(torch.tensor(sum(nlls) / n_tokens))
            return ppl.item()

    def fit(self, X_train, y_train):
        print(f"Using {self.device}")
        """Fit classifier by choosing optimal perplexity threshold."""
        perplexities = [self.compute_perplexity(text) for text in X_train]

        # Binary conversion of labels: human=0, ChatGPT=1
        y_binary = np.array([0 if y == 'human' else 1 for y in y_train])

        best_acc, best_thresh = 0, None
        thresholds = np.linspace(min(perplexities), max(perplexities), num=100)
        for thresh in thresholds:
            preds = np.array([1 if ppl < thresh else 0 for ppl in perplexities])
            acc = (preds == y_binary).mean()
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        self.threshold = best_thresh
        print(f"Optimal threshold set at {self.threshold:.2f} with training accuracy {best_acc:.4f}")
        return self

    def predict(self, X_test):
        """Predict labels based on computed perplexities."""
        if self.threshold is None:
            raise ValueError("Classifier not fitted. Call 'fit' first.")

        perplexities = [self.compute_perplexity(text) for text in X_test]
        predictions = ['ChatGPT' if ppl < self.threshold else 'human' for ppl in perplexities]

        return np.array(predictions)











class RoBERTaFinetunedClassifier:
    """RoBERTa-Finetuned classifier implementation (Liu et al., 2019)"""
    
    def __init__(self, model_name: str = 'roberta-base', 
                 batch_size: int = 16, 
                 learning_rate: float = 5e-5,
                 num_epochs: int = 1,
                 max_length: int = 512):
        """
        Initialize RoBERTa-Finetuned classifier
        
        Args:
            model_name: Name of the RoBERTa model to use
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(self.device)
        
    def fit(self, X_train: pd.Series, y_train: pd.Series) -> 'RoBERTaFinetunedClassifier':
        """
        Fine-tune RoBERTa model on the training data
        
        Args:
            X_train: Training texts
            y_train: Training labels ('human' or 'ChatGPT')
        """
        print("Preparing training data...")
        train_dataset = TextDataset(tolist_safe(X_train), tolist_safe(y_train), self.tokenizer, self.max_length)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir='./roberta_finetuned_results',
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="no",
        )
        
        # Set up trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        print("Fine-tuning RoBERTa model...")
        trainer.train()
        
        return self
    
    def predict(self, X_test: pd.Series) -> np.ndarray:
        """
        Make predictions using the fine-tuned RoBERTa model
        
        Args:
            X_test: Test texts
            
        Returns:
            Array of predictions ('human' or 'ChatGPT')
        """
        print("Preparing test data...")
        # Create dummy labels (not used for prediction)
        dummy_labels = ['human'] * len(X_test)
        test_dataset = TextDataset(tolist_safe(X_test), dummy_labels, self.tokenizer, self.max_length)
        
        # Use DataLoader for batching
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # Make predictions
        self.model.eval()
        predictions = []
        
        print("Making predictions...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                # Convert binary predictions to labels
                batch_predictions = ['ChatGPT' if pred == 1 else 'human' for pred in preds]
                predictions.extend(batch_predictions)
        
        return np.array(predictions)


class StyleFeatureExtractor:
    """Stylometric feature extractor for text data"""
    
    def __init__(self):
        """Initialize style feature extractor"""
        pass
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract stylometric features from texts
        
        Args:
            texts: List of text samples
            
        Returns:
            Array of stylometric features
        """
        features = []
        
        for text in texts:
            text = str(text)
            
            # Text length features
            char_count = len(text)
            word_count = len(text.split())
            avg_word_length = char_count / max(1, word_count)
            
            # Sentence features
            sentences = text.split('.')
            sentence_count = len([s for s in sentences if s.strip()])
            avg_sentence_length = word_count / max(1, sentence_count)
            
            # Punctuation features
            comma_count = text.count(',')
            period_count = text.count('.')
            question_mark_count = text.count('?')
            exclamation_mark_count = text.count('!')
            total_punctuation = comma_count + period_count + question_mark_count + exclamation_mark_count
            
            # Uppercase features
            uppercase_count = sum(1 for c in text if c.isupper())
            uppercase_ratio = uppercase_count / max(1, char_count)
            
            # Special character ratios
            digit_ratio = sum(1 for c in text if c.isdigit()) / max(1, char_count)
            whitespace_ratio = sum(1 for c in text if c.isspace()) / max(1, char_count)
            
            # Calculate feature vector
            feature_vector = [
                char_count,
                word_count,
                avg_word_length,
                sentence_count,
                avg_sentence_length,
                comma_count,
                period_count,
                question_mark_count,
                exclamation_mark_count,
                total_punctuation,
                uppercase_count,
                uppercase_ratio,
                digit_ratio,
                whitespace_ratio
            ]
            
            features.append(feature_vector)
        
        return np.array(features)


class RoBERTaStyloClassifier:
    """RoBERTa-Stylo classifier implementation (Kumarage et al., 2023)"""
    
    def __init__(self, model_name: str = 'roberta-base', 
                 batch_size: int = 16, 
                 learning_rate: float = 5e-5,
                 num_epochs: int = 3,
                 max_length: int = 512):
        """
        Initialize RoBERTa-Stylo classifier
        
        Args:
            model_name: Name of the RoBERTa model to use
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Initialize stylometric feature extractor
        self.style_extractor = StyleFeatureExtractor()
        
        # Define combined model (RoBERTa + stylometric features)
        self.model = RoBERTaStyloModel(model_name, num_style_features=14)
        self.model.to(self.device)
        
    def fit(self, X_train: pd.Series, y_train: pd.Series) -> 'RoBERTaStyloClassifier':
        """
        Train RoBERTa-Stylo model on the training data
        
        Args:
            X_train: Training texts
            y_train: Training labels ('human' or 'ChatGPT')
        """
        print("Extracting stylometric features...")
        style_features = self.style_extractor.extract_features(X_train.tolist())
        
        print("Preparing training data...")
        train_texts = X_train.tolist()
        train_labels = [1 if label == 'ChatGPT' else 0 for label in y_train]
        
        # Create dataset
        train_dataset = StyloDataset(
            texts=train_texts,
            labels=train_labels,
            style_features=style_features,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        print("Training RoBERTa-Stylo model...")
        self.model.train()
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in train_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                style_features = batch['style_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, style_features)
                
                # Calculate loss
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def predict(self, X_test: pd.Series) -> np.ndarray:
        """
        Make predictions using the trained RoBERTa-Stylo model
        
        Args:
            X_test: Test texts
            
        Returns:
            Array of predictions ('human' or 'ChatGPT')
        """
        print("Extracting stylometric features for test data...")
        style_features = self.style_extractor.extract_features(X_test.tolist())
        
        print("Preparing test data...")
        test_texts = X_test.tolist()
        test_labels = [0] * len(X_test)  # Dummy labels, not used for prediction
        
        # Create dataset
        test_dataset = StyloDataset(
            texts=test_texts,
            labels=test_labels,
            style_features=style_features,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Make predictions
        self.model.eval()
        predictions = []
        
        print("Making predictions...")
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                style_features = batch['style_features'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, style_features)
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # Convert binary predictions to labels
                batch_predictions = ['ChatGPT' if pred == 1 else 'human' for pred in preds]
                predictions.extend(batch_predictions)
        
        return np.array(predictions)


class RoBERTaStyloModel(nn.Module):
    """Combined RoBERTa + Stylometric features model"""
    
    def __init__(self, model_name: str, num_style_features: int = 14):
        """
        Initialize RoBERTa-Stylo model
        
        Args:
            model_name: Name of the RoBERTa model to use
            num_style_features: Number of stylometric features
        """
        super().__init__()
        
        # Load pre-trained RoBERTa model
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Remove the RoBERTa classification head
        roberta_hidden_size = self.roberta.config.hidden_size
        
        # Create new modules for the combined model
        self.style_projection = nn.Linear(num_style_features, 64)
        self.combined_layer = nn.Linear(roberta_hidden_size + 64, 256)
        self.classifier = nn.Linear(256, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask, style_features):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            style_features: Stylometric features
            
        Returns:
            Model outputs
        """
        # Get RoBERTa outputs
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        roberta_hidden = outputs.hidden_states[-1][:, 0, :]  # Use [CLS] token representation
        
        # Process stylometric features
        style_projected = self.relu(self.style_projection(style_features))
        
        # Combine RoBERTa and stylometric features
        combined = torch.cat([roberta_hidden, style_projected], dim=1)
        combined = self.dropout(combined)
        
        # Combined layer
        hidden = self.relu(self.combined_layer(combined))
        hidden = self.dropout(hidden)
        
        # Classification layer
        logits = self.classifier(hidden)
        
        return logits


class StyloDataset(Dataset):
    """Dataset class for RoBERTa-Stylo model"""
    
    def __init__(self, texts, labels, style_features, tokenizer, max_length=512):
        """
        Initialize dataset
        
        Args:
            texts: List of texts
            labels: List of labels (0 for human, 1 for ChatGPT)
            style_features: Numpy array of stylometric features
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.style_features = torch.tensor(style_features, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        style_feature = self.style_features[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'style_features': style_feature,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    



class TextDataset(Dataset):
    """Dataset class for text classification with transformers"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = 1 if self.labels[idx] == 'ChatGPT' else 0
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



class RoBERTaPretrainedDetector:
    """RoBERTa model pre-trained for AI text detection"""
    
    def __init__(self, model_name: str = 'openai-community/roberta-large-openai-detector', 
                batch_size: int = 16, max_length: int = 512):
        """
        Initialize pre-trained RoBERTa detector
        
        Args:
            model_name: Name of the pre-trained model for AI detection
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
    def fit(self, X_train: pd.Series, y_train: pd.Series) -> 'RoBERTaPretrainedDetector':
        """
        Dummy fit method (no actual training needed)
        
        Args:
            X_train: Training texts (not used)
            y_train: Training labels (not used)
        """
        print("Model is already trained for AI text detection. Skipping training.")
        return self
        
    def predict(self, X_test: pd.Series) -> np.ndarray:
        """
        Make predictions using the pre-trained model
        
        Args:
            X_test: Test texts
            
        Returns:
            Array of predictions ('human' or 'ChatGPT')
        """
        print("Making predictions with pre-trained model...")
        predictions = []
        
        # Process in batches
        for i in range(0, len(X_test), self.batch_size):
            batch_texts = X_test[i:i+self.batch_size].tolist()
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # For openai-community/roberta-large-openai-detector:
                # Label 0 = real (human), Label 1 = fake (AI)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                # Convert to expected format
                batch_predictions = ['ChatGPT' if pred == 1 else 'human' for pred in preds]
                predictions.extend(batch_predictions)
        
        return np.array(predictions)
    

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _delete_sentences(text: str, p: float = 0.25, min_keep: int = 1) -> str:
    """Randomly delete each sentence with probability *p* (paper uses 0.25)."""
    sents = _SENT_SPLIT.split(text.strip())
    if len(sents) <= min_keep:
        return text
    kept = [s for s in sents if random.random() > p] or random.sample(sents, k=min_keep)
    return " ".join(kept)

class _TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[np.ndarray], tokenizer, max_len: int,
                 augment: bool, drop_p: float):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
        self.augment, self.drop_p = augment, drop_p
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        txt = _delete_sentences(self.texts[idx], self.drop_p) if self.augment else self.texts[idx]
        enc = self.tokenizer(txt, truncation=True, padding='max_length', max_length=self.max_len,
                             return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ---------------------------------------------------------------------------
# PU-loss (Kiryo et al. 2017, non-negative form)
# ---------------------------------------------------------------------------

def _pu_loss(logits: torch.Tensor, y: torch.Tensor, prior: float) -> torch.Tensor:
    prob_pos = F.softmax(logits, dim=1)[:, 1]
    lp, ln = F.binary_cross_entropy(prob_pos, torch.ones_like(prob_pos), reduction='none'), \
             F.binary_cross_entropy(prob_pos, torch.zeros_like(prob_pos), reduction='none')
    mask_p, mask_u = y == 1, y == 0
    p_loss = lp[mask_p].mean() if mask_p.any() else 0.
    n_loss_p = ln[mask_p].mean() if mask_p.any() else 0.
    n_loss_u = ln[mask_u].mean() if mask_u.any() else 0.
    risk = prior * p_loss + (n_loss_u - prior * n_loss_p)
    return torch.clamp(risk, min=0.)

# ---------------------------------------------------------------------------
# MPU Classifier
# ---------------------------------------------------------------------------

class MPUClassifier:
    """Hugging Face model-based implementation of MPU Classifier"""
    
    def __init__(self, model_name: str = 'yuchuantian/AIGC_detector_env2', 
                 max_len: int = 512, 
                 batch_size: int = 32,
                 device: Optional[str] = None, 
                 verbose: bool = True):
        """
        Initialize the MPU classifier using a pre-trained Hugging Face model
        
        Args:
            model_name: Name of the Hugging Face model (default is YuchuanTian/AIGC_text_detector)
            max_len: Maximum sequence length
            batch_size: Batch size for inference
            device: Device to use (cuda or cpu)
            verbose: Whether to print verbose output
        """
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        if self.verbose:
            print(f"Loading pre-trained MPU model from {model_name}")
            print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model directly from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
    def fit(self, X: List[str], y_labels: List[str]):
        """
        Dummy fit method (no training needed for pre-trained model)
        
        Args:
            X: Training texts (not used)
            y_labels: Training labels (not used)
        """
        if self.verbose:
            print(f"Using pre-trained MPU model. No fitting required.")
        return self
        
    @torch.no_grad()
    def predict(self, X: List[str]) -> List[str]:
        """
        Make predictions using the pre-trained model
        
        Args:
            X: Test texts
            
        Returns:
            List of predictions ('human' or 'ChatGPT')
        """
        if isinstance(X, (pd.Series, np.ndarray)):
            X = X.tolist()
            
        self.model.eval()
        predictions = []
        
        # Process in batches
        for i in range(0, len(X), self.batch_size):
            batch_texts = X[i:i+self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # For MPU models, typically: 0 = AI, 1 = human
            # But verify this with specific model documentation
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Convert to expected format
            batch_predictions = ['ChatGPT' if pred == 1 else 'human' for pred in pred_labels]
            predictions.extend(batch_predictions)
            
            if self.verbose and i % 100 == 0:
                print(f"Processed {i}/{len(X)} texts")
                
        return predictions
