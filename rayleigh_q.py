import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import argparse
from sklearn.feature_extraction.text import CountVectorizer

# Argument parser for command-line inputs
def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute Rayleigh Quotient for Text Data")
    parser.add_argument('--mode', type=str, default='word_count', 
                        help="Choose 'word_count' for Bag-of-Words or 'embeddings' for sentence embeddings (default)")
    return parser.parse_args()

# Function to compute Rayleigh quotient
def rayleigh_quotient(v, labels):
    # Compute class means (human and ChatGPT) and overall mean
    classes = np.unique(labels)
    class_means = np.array([np.mean(v[labels == c]) for c in classes])
    overall_mean = np.mean(v)
    
    # Compute between-class variance
    between_class_variance = np.sum([len(v[labels == c]) * (class_mean - overall_mean)**2 
                                     for c, class_mean in zip(classes, class_means)])
    
    # Compute within-class variance
    within_class_variance = np.sum([np.sum((v[labels == c] - class_mean)**2) 
                                    for c, class_mean in zip(classes, class_means)])
    
    # Compute total variance
    total_variance = np.sum((v - overall_mean)**2)
    
    # Compute normalized Rayleigh quotient
    return 1 - (within_class_variance / total_variance)

# Main function to handle the logic
def main():
    args = parse_arguments()
    
    # Load the data
    df = pd.read_csv('ades.csv')
    labels = np.array(['human'] * len(df['human']) + ['ChatGPT'] * len(df['ChatGPT']))

    if args.mode == 'word_count':
        print("Using Bag-of-Words representation.")
        
        # Combine 'human' and 'ChatGPT' texts for Bag-of-Words
        corpus = df['human'].tolist() + df['ChatGPT'].tolist()
        
        # Create Bag-of-Words matrix
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus).toarray()
        print(f"Shape of Bag-of-Words matrix: {X.shape}")
    
    else:
        print("Using sentence embeddings.")
        
        # Compute embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        human_embeddings = model.encode(df['human'].tolist())
        chatgpt_embeddings = model.encode(df['ChatGPT'].tolist())
        
        # Combine embeddings
        X = np.vstack((human_embeddings, chatgpt_embeddings))
        print(f"Shape of embeddings matrix: {X.shape}")

    # Compute X X^T
    XXT = np.dot(X, X.T)

    # Compute eigenvalues and eigenvectors of X X^T
    eigenvalues, eigenvectors = np.linalg.eigh(XXT)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    print("eigenvalues = ", eigenvalues[:20])  # Print first 20 eigenvalues
    eigenvectors = eigenvectors[:, idx]

    # Select top K eigenvectors (you can adjust K as needed)
    K = 20
    top_eigenvectors = eigenvectors[:, :K]

    # Compute Rayleigh quotient for each top eigenvector projection
    rayleigh_quotients = []
    for ev in top_eigenvectors.T:
        rq = rayleigh_quotient(ev, labels)
        rayleigh_quotients.append(rq)

    # Plot Rayleigh quotients
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, K+1), rayleigh_quotients, 'bo-')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Rayleigh Quotient')
    plt.title(f'Rayleigh Quotient vs Eigenvalue Index using {args.mode}')
    plt.grid(True)
    plt.savefig(f'rayleigh_quotient_plot_{args.mode}.png')
    plt.show()

    # Plot eigenvalues
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, 'ro-')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues of X X^T')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'eigenvalues_plot_{args.mode}.png')
    plt.show()

# Entry point of the script
if __name__ == "__main__":
    main()
