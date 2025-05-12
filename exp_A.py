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



import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np

from Classifiers.classifiers_class import (
    TFIDFClassifier, 
    EmbeddingsRandomForestClassifier,
    TopicScoreClassifier,
    PHCClassifier,
    DetectGPTClassifier,
    PHCLLMClassifier,
    TopicScoreLLMClassifier,
    RoBERTaStyloClassifier,
    RoBERTaFinetunedClassifier,
    TextClassifier,
    MPUClassifier
)
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

# class CUDRTJSONLoader(TextDataLoader):
#     def __init__(self, json_path: str, type_filter: str = None):
#         super().__init__()
#         self.json_path = json_path
#         self.type_filter = type_filter

#     def load_data(self) -> Tuple[pd.DataFrame, List[str]]:
#         print(f"Loading JSON from {self.json_path}")
#         with open(self.json_path, "r", encoding="utf-8") as f:
#             raw = json.load(f)

#         # Apply filter if specified
#         if self.type_filter:
#             raw = [entry for entry in raw if entry["Type"] == self.type_filter]

#         texts = []
#         labels = []
#         for entry in raw:
#             texts.extend([entry["Human_Content"].strip(), entry["AI_Content"].strip()])
#             labels.extend(["human", "ChatGPT"])

#         self.labels = labels
#         self.data = pd.DataFrame({"text": texts, "label": labels})
#         print(f"Loaded {len(self.data)} samples with filter = {self.type_filter}")
#         return self.data, self.labels

#     def get_label_pairs(self) -> List[Tuple[str, str]]:
#         if self.labels is None:
#             self.load_data()
#         return [("human", "ChatGPT")]

class CUDRTJSONLoader(TextDataLoader):
    def __init__(self, json_path: str, type_filter: str = None):
        super().__init__()
        self.json_path = json_path
        self.type_filter = type_filter

    def load_full_dataframe(self) -> pd.DataFrame:
        """Load as DataFrame with human + AI columns (before flattening)."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if self.type_filter:
            raw = [entry for entry in raw if entry["Type"] == self.type_filter]

        data = []
        for entry in raw:
            data.append({
                "Human_Content": entry["Human_Content"].strip(),
                "AI_Content": entry["AI_Content"].strip()
            })

        return pd.DataFrame(data)

    def load_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Original flattened long format (default fallback)."""
        df = self.load_full_dataframe()
        texts = list(df["Human_Content"]) + list(df["AI_Content"])
        labels = ["human"] * len(df) + ["ChatGPT"] * len(df)
        return pd.DataFrame({"text": texts, "label": labels}), labels

    def get_label_pairs(self) -> List[Tuple[str, str]]:
        if self.labels is None:
            self.load_data()
        return [("human", "ChatGPT")]




def run_comparison(data_loader: TextDataLoader,
                  methods: Dict[str, TextClassifier],
                  cv_folds: int = 5,
                  test_size: float = 0.2,
                  use_cv=False) -> Dict:
    """Run comparison between multiple classification methods"""
    results = {}

    # Handle pair splitting for CSV and JSON loaders
    if isinstance(data_loader, StandardCSVDataLoader):
        df = pd.read_csv(data_loader.csv_path)
        human_col = data_loader.human_col
        ai_col = data_loader.ai_col

        # Split by row to preserve pairs
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=201
        )

        X_train = pd.concat([train_df[human_col], train_df[ai_col]])
        y_train = ['human'] * len(train_df) + ['ChatGPT'] * len(train_df)

        X_test = pd.concat([test_df[human_col], test_df[ai_col]])
        y_test = ['human'] * len(test_df) + ['ChatGPT'] * len(test_df)

    elif isinstance(data_loader, CUDRTJSONLoader):
        df = data_loader.load_full_dataframe()

        # Split by row to preserve pairs
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=40
        )

        X_train = pd.concat([train_df['Human_Content'], train_df['AI_Content']])
        y_train = ['human'] * len(train_df) + ['ChatGPT'] * len(train_df)

        X_test = pd.concat([test_df['Human_Content'], test_df['AI_Content']])
        y_test = ['human'] * len(test_df) + ['ChatGPT'] * len(test_df)

    else:
        # Fallback: generic loaders
        data, labels = data_loader.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            data['text'], data['label'],
            test_size=test_size,
            stratify=data['label'],
            random_state=4
        )

    # Run each method
    for method_name, classifier in methods.items():
        print(f"\nRunning {method_name}...")
        try:
            if method_name == 'TopicScore' and use_cv:
                # Define parameter grid
                n_topics_values = [3, 4, 5]
                threshold_values = [0.001, 0.01]
                
                best_accuracy = 0
                best_params = None
                
                # Grid search with cross-validation
                for n_topics in n_topics_values:
                    for threshold in threshold_values:
                        print(f"\nTrying n_topics={n_topics}, threshold={threshold}")
                        cv_accuracies = []
                        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        
                        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                            X_train_fold = X_train.iloc[train_idx]
                            y_train_fold = np.array(y_train)[train_idx]
                            X_val_fold = X_train.iloc[val_idx]
                            y_val_fold = np.array(y_train)[val_idx]
                            
                            fold_classifier = TopicScoreClassifier(n_topics=n_topics, threshold=threshold)
                            fold_classifier.fit(X_train_fold, y_train_fold)
                            
                            val_predictions = fold_classifier.predict(X_val_fold)
                            fold_accuracy = accuracy_score(y_val_fold, val_predictions)
                            cv_accuracies.append(fold_accuracy)
                        
                        mean_cv_accuracy = np.mean(cv_accuracies)
                        print(f"Mean CV accuracy: {mean_cv_accuracy:.4f}")
                        
                        if mean_cv_accuracy > best_accuracy:
                            best_accuracy = mean_cv_accuracy
                            best_params = {'n_topics': n_topics, 'threshold': threshold}
                
                print(f"\nBest parameters found: {best_params}")
                print(f"Best CV accuracy: {best_accuracy:.4f}")
                
                best_classifier = TopicScoreClassifier(**best_params)
                best_classifier.fit(X_train, y_train)
                predictions = best_classifier.predict(X_test)
                
            else:
                # Standard fit + predict
                classifier.fit(X_train, y_train)
                predictions = classifier.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)
            
            results[method_name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'predictions': predictions
            }
            
            if method_name == 'TopicScore' and use_cv:
                results[method_name]['best_parameters'] = best_params
                results[method_name]['best_cv_accuracy'] = best_accuracy
            
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
    parser.add_argument('--data-format', choices=['coauthor', 'csv', 'cudrt_json'], required=True)
    parser.add_argument('--type-filter', default=None, help="Only load JSON entries with this 'Type' value")
    parser.add_argument('--llm-model', default='gpt-4o-mini')
    # ---- in the main() block, extend the argument parser ----
    parser.add_argument(
        '--ai-col',
        default='ChatGPT',
        help='Name of the CSV column that contains the AI-edited text '
            '(default: ChatGPT). Labels remain "human" vs "ChatGPT".'
    )

    args = parser.parse_args()
    

    # Choose loader
    if args.data_format == 'coauthor':
        data_loader = CoAuthorDataLoader(args.data_path)
    elif args.data_format == 'csv':
        data_loader = StandardCSVDataLoader(args.data_path, human_col='human', ai_col=args.ai_col)
    elif args.data_format == 'cudrt_json':
        data_loader = CUDRTJSONLoader(json_path=args.data_path, type_filter=args.type_filter)

    
    # Initialize all classifiers
    methods = {
        #'TopicScore': TopicScoreClassifier(n_topics=5, threshold=0.001),
        #'TF-IDF': TFIDFClassifier()
        # 'PHC': PHCClassifier(),
        'MPU': MPUClassifier()
        # 'RoBERTa-finetuned':RoBERTaFinetunedClassifier(),
        # 'RoBERTa-stylo':RoBERTaStyloClassifier(),
        # 'DetectGPT':DetectGPTClassifier(),
        # 'LLM': PHCLLMClassifier(model_name=args.llm_model, method="llm_only"),
        # 'LLM+PHC': PHCLLMClassifier(model_name=args.llm_model, method="llm_phc")
        #  'TopicScoreLLM': TopicScoreLLMClassifier(
        # model_name=args.llm_model,
        # n_topics=5,
        # threshold=0.001,
        # method="llm_ts")
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
