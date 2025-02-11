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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import traceback
from Classifiers.PHC import get_pvals as phc_get_pvals, accuracy as phc_accuracy, clean_text as phc_clean_text
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# from utils import *
from utils_2 import *

# Configuration
MODELS = ['llama', 'claude-3-5-sonnet', 'deepseek-reasoner', 'gpt-4o-mini', 'gemini']
N_FOLDS = 5
ADDITIONAL_STOPS = ['Abstract']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def run_phc(train_df, test_df, vocabulary):
    try:
        phc_train = pd.DataFrame({
            'author': ['human']*len(train_df) + ['ai']*len(train_df),
            'text': list(train_df['human']) + list(train_df['ChatGPT'])
        })
        
        phc_test = pd.DataFrame({
            'author': ['human']*len(test_df) + ['ai']*len(test_df),
            'text': list(test_df['human']) + list(test_df['ChatGPT'])
        })
        
        phc_train_clean = phc_clean_text(phc_train)
        phc_test_clean = phc_clean_text(phc_test)
        
        phc_preds, phc_true, selected_words_info = phc_get_pvals(
            phc_train_clean, 
            phc_test_clean, 
            show_hist=False,
            vocab=vocabulary
        )
        
        
        return ['human' if p == 1 else 'ChatGPT' for p in phc_preds], phc_true, selected_words_info
    except Exception as e:
        print(f"PHC Error: {str(e)}")
        return None, None, None


def process_author(author, df, output_dir="results_edited"):
    author_data = df[df['author'] == author].copy()
    if len(author_data) < 20:
        return None, None
    
    all_predictions = []
    all_thoughts = []
    all_phc_words = []
    
    for fold in range(N_FOLDS):
        try:
            # Train-test split
            train_df, test_df = train_test_split(author_data, test_size=0.2, random_state=fold)
            
            # Create vocabulary
            stop_words = list(ENGLISH_STOP_WORDS) + ADDITIONAL_STOPS
            vectorizer = CountVectorizer(
                stop_words=stop_words,
                min_df=5,
                max_df=0.95,
                token_pattern=r'(?u)\b\w+\b'
            )
            train_texts = train_df['human'].tolist() + train_df['ChatGPT'].tolist()
            train_texts = [' '.join([word for word in t.split() if len(word) >= 5 and not word.isdigit()]) 
                          for t in train_texts]
            vectorizer.fit(train_texts)
            vocabulary = vectorizer.get_feature_names_out()

            # Prepare test data
            test_texts = list(test_df['human']) + list(test_df['ChatGPT'])
            true_labels = ['human']*len(test_df) + ['ChatGPT']*len(test_df)
            
            # Run PHC
            phc_preds, _, phc_words = run_phc(train_df, test_df, vocabulary)
            if phc_words:
                # Sort words by effect size in descending order
                sorted_words = sorted(phc_words, key=lambda x: x['effect_size'], reverse=True)
                top_15_words = [(word['word'], word['direction']) for word in sorted_words[:15]]
                
                # Store the top words with fold and author information
                all_phc_words.append({
                    'author': author,
                    'fold': fold,
                    'top_15_words': top_15_words
                })
            
            # Collect predictions for this fold
            fold_predictions = []
            for i, (text, true_label) in enumerate(zip(test_texts, true_labels)):
                record = {
                    'author': author,
                    'fold': fold,
                    'text': text,
                    'true_label': true_label,
                    'phc_pred': phc_preds[i] if phc_preds else None
                }
                fold_predictions.append(record)
            
            # Run LLM models
            for model in MODELS:
                try:
                    # Pure LLM
                    
                    pure_results = llm_classifier(test_texts,model= model)
                    # PHC-assisted LLM
                    phc_results = llm_classifier(test_texts, model=model, selected_words_info=phc_words)
                    
                    for i, record in enumerate(fold_predictions):
                        record[f'{model}_pure_pred'] = pure_results['predictions'][i]
                        record[f'{model}_phc_pred'] = phc_results['predictions'][i]
                        
                        # Capture DeepSeek thoughts
                        if model == 'deepseek-reasoner':
                            all_thoughts.append({
                                'author': author,
                                'fold': fold,
                                'text': test_texts[i],
                                'thoughts_pure': pure_results['thoughts'][i],
                                'thoughts_phc': phc_results['thoughts'][i]
                            })
                except Exception as e:
                    print(f"Error with {model}: {str(e)}")
            
            all_predictions.extend(fold_predictions)
            
        except Exception as e:
            print(f"Fold {fold} error: {str(e)}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    prediction_df = pd.DataFrame(all_predictions)
    prediction_path = os.path.join(output_dir, f"{author}_predictions.csv")
    prediction_df.to_csv(prediction_path, index=False)

    # Save PHC words
    phc_words_df = pd.DataFrame(all_phc_words)
    phc_words_path = os.path.join(output_dir, f"{author}_phc_words.csv")
    phc_words_df.to_csv(phc_words_path, index=False)
    
    thoughts_df = pd.DataFrame(all_thoughts)
    return prediction_df, thoughts_df

if __name__ == "__main__":
    df = pd.read_csv('data_edited.csv')
    for col in ['human', 'ChatGPT']:
        if col in df.columns:
            # Remove specific phrases and non-alphabetic characters
            df[col] = df[col].apply(lambda x: re.sub("All rights", "", str(x)))
            df[col] = df[col].apply(lambda x: re.sub("reserved", "", str(x)))
            df[col] = df[col].apply(lambda x: re.sub("[^A-Za-z ]", "", str(x)))
            df[col] = df[col].apply(lambda x: re.sub("copyright", "", str(x)))
            # Additional cleanup
            df[col] = df[col].apply(lambda x: ' '.join(x.split()))  # Remove extra whitespac
    selected_authors = pd.read_csv("single_GPT.csv")["author"].unique().tolist()
    
    all_thoughts = []
    
    for author in selected_authors:
        print(f"\nProcessing author: {author}")
        try:
            _, author_thoughts = process_author(author, df)
            if author_thoughts is not None:
                all_thoughts.append(author_thoughts)
        except Exception as e:
            print(f"Error processing {author}: {str(e)}")
    
    # Save all DeepSeek thoughts
    if all_thoughts:
        final_thoughts = pd.concat(all_thoughts)
        final_thoughts.to_csv("deepseek_thoughts_edited.csv", index=False)