from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import os
import re
from openai import OpenAI
import time
from Classifiers.PHC import get_pvals as phc_get_pvals, accuracy as phc_accuracy, clean_text as phc_clean_text, get_vocab
import google.generativeai as genai
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM


def sentence_transformer_random_forest(df_train, 
                                    df_test,
                                    model_name = 'all-MiniLM-L6-v2',
                                    human_col = 'human',
                                    ai_col = 'ChatGPT',
                                    n_estimators = 100,
                                    max_features = 'sqrt',
                                    batch_size = 32,
                                    device = None) -> dict:
    
    # Automatically select device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Initialize the Sentence Transformer model
    print(f"Loading {model_name} model...")
    model = SentenceTransformer(model_name, device=device)
    
    # Prepare training data
    X_train = pd.concat([df_train[human_col], df_train[ai_col]])
    y_train = ['human'] * len(df_train) + ['ChatGPT'] * len(df_train)
    
    # Prepare test data
    X_test = pd.concat([df_test[human_col], df_test[ai_col]])
    y_test = ['human'] * len(df_test) + ['ChatGPT'] * len(df_test)
    
    # Compute embeddings for training data
    print("Computing embeddings for training data...")
    X_train_embeddings = model.encode(
        X_train.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        device=device
    )
    
    # Compute embeddings for test data
    print("Computing embeddings for test data...")
    X_test_embeddings = model.encode(
        X_test.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        device=device
    )
    
    # Train Random Forest classifier
    print("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    rf.fit(X_train_embeddings, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = rf.predict(X_test_embeddings)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature_index': range(X_train_embeddings.shape[1]),
        'importance': rf.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Calculate prediction probabilities
    y_prob = rf.predict_proba(X_test_embeddings)
    
    # Add confidence scores
    confidence_scores = np.max(y_prob, axis=1)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'feature_importance': feature_importance,
        'model': rf,
        'embeddings_model': model,
        'predictions': y_pred,
        'prediction_probabilities': y_prob,
        'confidence_scores': confidence_scores,
        'embedding_dimension': X_train_embeddings.shape[1]
    }


def extract_llm_answer(chat_completion) -> str:
    # Get the reasoning content from the first choice
    content = chat_completion.choices[0].message.reasoning_content.lower()
    
    # Find last position of each word
    human_pos = content.rfind('human')
    chatgpt_pos = content.rfind('chatgpt')
    
    # Return whichever appears last
    if human_pos > chatgpt_pos:
        return 'human'
    elif chatgpt_pos > human_pos:
        return 'ChatGPT'
    else:
        return 'No answer found'





def extract_llm_answer_new(chat_completion):
    # Get the content from the first choice
    content = chat_completion.choices[0].message.content
    
    # Extract thoughts (content between <think> tags)
    think_start = content.find('<think>') + len('<think>')
    think_end = content.find('</think>')
    thoughts = content[think_start:think_end].strip() if think_start > -1 and think_end > -1 else ""
    
    # Look for prediction in the last 100 characters of thoughts
    # (since the conclusion typically appears at the end)
    search_text = thoughts[-100:].lower()
    
    # Find last position of each word
    human_pos = search_text.rfind('human')
    chatgpt_pos = search_text.rfind('chatgpt')
    
    # Determine prediction based on which appears last
    if human_pos > chatgpt_pos and human_pos != -1:
        prediction = 'human'
    elif chatgpt_pos > human_pos and chatgpt_pos != -1:
        prediction = 'ChatGPT'
    else:
        prediction = 'No prediction found'
    
    return thoughts, prediction


def llm_classifier(test_documents, selected_words_info=None, api_key=None, base_url='https://api.deepseek.com',
                 model="llama3.1-70b", max_retries=10, delay_between_calls=0.1, human_words=None, ai_words=None):
    """
    A classifier that determines whether text is human-written or AI-generated using various LLM APIs.
    Updated to use Llama through the OpenAI-style API interface.
    """
    
    thoughts = []
    # Configure appropriate client based on model
    if model == "gemini":
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    elif model in ["claude-3-5-sonnet", "claude-3-haiku-20240307"]:
        if api_key is None:
            api_key = os.getenv('CLAUDE_API_KEY')
            if api_key is None:
                raise ValueError("No API key provided and ANTHROPIC_API_KEY not found in environment")
        claude_client = Anthropic(api_key=api_key)
    elif model.startswith("llama") or 'reasoner' in model:
        base_url = "https://api.llama-api.com"
        # Configure Llama client using OpenAI interface
        if api_key is None:
            api_key = os.getenv('LLAMA_API_KEY')
            if api_key is None:
                raise ValueError("No API key provided and LLAMA_API_KEY not found in environment")
        llama_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    elif "gpt" in model:
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("No API key provided and OPENAI_API_KEY not found in environment")
        client = OpenAI(api_key=api_key)

    # elif "reasoner" in model:
    #     client = OpenAI(
    #     base_url = "https://integrate.api.nvidia.com/v1",
    #     api_key = os.getenv("NVIDIA_API_KEY")
    #     )
        # print("all gut")
        # print(time.sleep(20))
    else:
        if api_key is None:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if api_key is None:
                raise ValueError("No API key provided and OPENAI_API_KEY not found in environment")
        client = OpenAI(api_key=api_key, base_url=base_url)

    predictions = []
    confidence_scores = []
   
    # Process discriminative words information
    if selected_words_info is not None:
        human_words = [info['word'] for info in selected_words_info 
                      if info['direction'] == 'Human'][:15]
        ai_words = [info['word'] for info in selected_words_info 
                   if info['direction'] == 'ChatGPT'][:15]
        
        discriminative_info = (
            f"\n\nBased on statistical analysis, here are characteristic words:"
            f"\nTypically human-written: {', '.join(human_words)}"
            f"\nTypically AI-written: {', '.join(ai_words)}"
        )
    elif human_words is not None and ai_words is not None:
        human_words = sorted(human_words, key=lambda x: -x[1])
        ai_words = sorted(ai_words, key=lambda x: -x[1])
        
        human_words = [word[0] for word in human_words][:10]
        ai_words = [word[0] for word in ai_words][:10]
        discriminative_info = (
            f"\n\nBased on statistical analysis, here are characteristic words:"
            f"\nTypically human-written: {', '.join(human_words)}"
            f"\nTypically AI-written: {', '.join(ai_words)}"
        )
    else:
        discriminative_info = ""
   
    # Define system prompt
    system_prompt = f"""You are a classifier that determines whether text is human-written or written by an AI. 
    You will be given academic abstracts that are either written by humans or by AI.{discriminative_info}
    
    Respond with exactly one word: either 'human' for human-written text or 'ChatGPT' for AI-edited text. 
    Be as accurate as possible in your classification, considering both the writing style and the presence of characteristic words."""
   
    # Process each document
    responses = []
    for i, doc in enumerate(test_documents):
        if i % 10 == 0:
            print(f"Processing document {i+1}/{len(test_documents)}")
            if model=='gemini':
                print("sleeping for 10s")
                time.sleep(30)
            
        for attempt in range(max_retries):
            try:
                if model == "gemini":
                    prompt = f"Here is your task: {system_prompt}\n\nClassify this text: {doc}"
                    response = gemini_model.generate_content(prompt)
                    prediction = response.text.strip().lower()
                
                elif model in ["claude-3-5-sonnet", "claude-3-haiku-20240307"]:
                    messages = [{
                        "role": "user",
                        "content": f"Here is your task: {system_prompt}\n\nClassify this text: {doc}"
                    }]
                    
                    response = claude_client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        messages=messages,
                        max_tokens=1,
                        temperature=0.1
                    )
                    
                    prediction = response.content[0].text.strip().lower()
                
                elif model.startswith("llama"):
                    model = "llama3.1-8b"
                    # Use OpenAI-style interface for Llama
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Classify this text: {doc}"}
                    ]
                    
                    response = llama_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.1,  # Adjusted temperature for slightly varied outputs
                        max_tokens=1,     # We only need one word
                        stream=False      # No streaming needed for classification
                    )
                    
                    prediction = response.choices[0].message.content.strip().lower()
                
                elif 'gpt' in model:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Classify this text: {doc}"}
                    ]
                    
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=1,
                        stream=False
                    )
                    # print("response = ",response)
                    responses.append(response)
                    #print(response)
                    #prediction = extract_llm_answer(response)
                    prediction = response.choices[0].message.content.strip().lower()
                

                elif "reasoner" in model:
                    messages = [
                        {"role": "user", "content": f"{system_prompt}. Classify this text: {doc}"}
                    ]
                    
                    # response = client.chat.completions.create(
                    #     model="deepseek-ai/deepseek-r1",
                    #     messages=messages,
                    #     temperature=0.6,
                    #     # max_tokens=250,
                    #     stream=False
                    # )
                    response = llama_client.chat.completions.create(
                        model="deepseek-r1",
                        messages=messages,
                        temperature=0.6,
                        # max_tokens=250,
                        stream=False
                    )
                    #print("response = ",response)
                    responses.append(response)
                    #print(response)
                    thought, prediction = extract_llm_answer_new(response)
                    thoughts.append(thought)
                    #print("prediction = ",prediction)

                else:
                    # OpenAI API logic
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Think fast (no long answer or thoughts just the answer) and finish with you prediction, human or ChatGPT. Classify this text: {doc}"}
                    ]
                    
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.5,
                        max_tokens=1,
                        stream=False
                    )
                    #print("response = ",response)


                    prediction = extract_llm_answer(response)
                    # thoughts.append(thought)
                    #prediction = response.choices[0].message.content.strip().lower()
                
                print("prediction = ", prediction)
                
                if prediction not in ['human', 'chatgpt']:
                    print(f"Warning: Unexpected prediction '{prediction}', defaulting to 'ChatGPT'")
                    prediction = 'ChatGPT'
                
                predictions.append(prediction)
                confidence_scores.append(None)
                
                time.sleep(delay_between_calls)
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error after {max_retries} attempts: {str(e)}")
                    predictions.append('ChatGPT')
                    confidence_scores.append(None)
                else:
                    time.sleep(delay_between_calls * (attempt + 1))
                    continue
    
    if "reasoner" in model:
        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'thoughts':thoughts
        }
    
    else:
        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores
        }
    

























# def llm_classifier2(test_documents: list[str], 
#                 api_key: str = None,
#                 model: str = "gpt-4-mini",
#                 max_retries: int = 3,
#                 delay_between_calls: float = 1.0):
#     if api_key is None:
#         api_key = os.getenv('OPENAI_API_KEY')
#         if api_key is None:
#             raise ValueError("No API key provided and OPENAI_API_KEY not found in environment")

#     client = OpenAI(api_key=api_key)
#     predictions = []
#     confidence_scores = []

#     # system_prompt = """You are a classifier that determines whether text is human-written or AI-edited. 
#     # Respond with exactly one word: either 'human' for human-written text or 'ChatGPT' for AI-edited text. 
#     # Be as accurate as possible in your classification."""
#     system_prompt = """You are a classifier that determines whether text is human-written or AI-edited. 
#     Respond with exactly one word: either 'human' for human-written text or 'ChatGPT' for AI written text. 
#     Be as accurate as possible in your classification."""

#     for doc in test_documents:
#         for attempt in range(max_retries):
#             try:
#                 messages = [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": f"Classify this text: {doc}"}
#                 ]
                
#                 # Create streaming completion
#                 stream = client.chat.completions.create(
#                     model=model,
#                     messages=messages,
#                     temperature=0,  # Use deterministic output
#                     max_tokens=10,  # We only need a single word response
#                     stream=True
#                 )
                
#                 # Collect the streamed response
#                 collected_response = ""
#                 for chunk in stream:
#                     if chunk.choices[0].delta.content is not None:
#                         collected_response += chunk.choices[0].delta.content
                
#                 # Extract the classification
#                 prediction = collected_response.strip().lower()
                
#                 # Ensure the prediction matches our expected labels
#                 if prediction not in ['human', 'chatgpt']:
#                     print(f"Warning: Unexpected prediction '{prediction}', defaulting to 'ChatGPT'")
#                     prediction = 'ChatGPT'
                
#                 predictions.append(prediction)
#                 confidence_scores.append(None)  # No confidence score in streaming response
                
#                 # Add delay between calls to respect rate limits
#                 time.sleep(delay_between_calls)
#                 break
                
#             except Exception as e:
#                 if attempt == max_retries - 1:
#                     print(f"Error after {max_retries} attempts: {str(e)}")
#                     predictions.append('ChatGPT')  # Default prediction on error
#                     confidence_scores.append(None)
#                 else:
#                     time.sleep(delay_between_calls * (attempt + 1))  # Exponential backoff
#                     continue

#     return {
#         'predictions': predictions,
#         'confidence_scores': confidence_scores
#     }