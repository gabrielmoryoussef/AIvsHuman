# import numpy as np
# from scipy.sparse import csr_matrix
# from scipy.sparse.linalg import svds
# from sklearn.cluster import KMeans
# from scipy.optimize import minimize
# import warnings
# import itertools
# def topic_score(K: int, X: np.ndarray, K0: int = None, m: int = None, 
#                 Mquantile: float = 0, num_restart: int = 1, seed: int = None) -> dict:
#     """
#     Implementation of Topic SCORE algorithm with improved numerical stability.
    
#     Parameters:
#     -----------
#     K : int
#         Number of topics
#     X : np.ndarray
#         Word-document matrix (p x n)
#     K0 : int, optional
#         Number of greedy search steps
#     m : int, optional
#         Number of centers in kmeans
#     Mquantile : float
#         Quantile for truncation
#     num_restart : int
#         Number of restarts for kmeans
#     seed : int, optional
#         Random seed
    
#     Returns:
#     --------
#     dict
#         Results including A_hat, R, V, Pi, theta
#     """
#     if seed is not None:
#         np.random.seed(seed)
    
#     if K0 is None:
#         K0 = int(np.ceil(1.5 * K))
#     if m is None:
#         m = 10 * K
    
#     p, n = X.shape
    
#     # Compute row means and handle truncation
#     M = np.mean(X, axis=1)
#     M_trunk = np.minimum(M, np.quantile(M, Mquantile)) if Mquantile > 0 else M
    
#     # Add small constant to avoid division by zero
#     epsilon = 1e-10
#     M_trunk = M_trunk + epsilon
    
#     # Normalize and compute SVD
#     X_scaled = np.diag(np.sqrt(1/M_trunk)) @ X
#     U, _, _ = svds(X_scaled, K)
#     Xi = np.fliplr(U)  # Reverse columns to match R's behavior
    
#     # Step 1: Calculate ratios with zero handling
#     Xi[:, 0] = np.abs(Xi[:, 0])  # Take absolute value of first column
    
#     # Handle zeros in the denominator
#     mask = Xi[:, 0] > epsilon
#     R = np.zeros((p, K-1))
#     R[mask] = (Xi[mask, 1:K] / Xi[mask, 0:1]).reshape(-1, K-1)
    
#     # Replace any remaining NaNs or infs with zeros
#     R = np.nan_to_num(R, 0)
    
#     # Step 2: Find vertices
#     V, theta = vertices_est(R, K0, m, num_restart)
    
#     # Step 3: Calculate convex combinations
#     augmented_R = np.column_stack([R, np.ones(p)])
#     augmented_V = np.column_stack([V, np.ones(K)])
    
#     try:
#         Pi = augmented_R @ np.linalg.inv(augmented_V)
#     except np.linalg.LinAlgError:
#         # If matrix is singular, use pseudo-inverse
#         Pi = augmented_R @ np.linalg.pinv(augmented_V)
    
#     # Ensure non-negative values and normalize
#     Pi = np.maximum(Pi, 0)
#     Pi = Pi / (Pi.sum(axis=1, keepdims=True) + epsilon)
    
#     # Step 4: Calculate word-topic matrix
#     A_hat = np.sqrt(M_trunk)[:, np.newaxis] * Xi[:, 0:1] * Pi
    
#     # Step 5: Normalize
#     A_hat = A_hat / (A_hat.sum(axis=0) + epsilon)
    
#     return {
#         'A_hat': A_hat,
#         'R': R,
#         'V': V,
#         'Pi': Pi,
#         'theta': theta
#     }

# def simplex_dist(theta: np.ndarray, V: np.ndarray) -> float:
#     """
#     Compute L2 distance between a point and a simplex.
    
#     Parameters:
#     -----------
#     theta : np.ndarray
#         Point of shape (K-1,)
#     V : np.ndarray
#         Matrix of vertices of shape (K, K-1)
    
#     Returns:
#     --------
#     float
#         L2 distance
#     """
#     # V is K x (K-1), we need first K-1 rows for difference matrix
#     V_diff = V[:-1] - V[-1]  # This is (K-1) x (K-1)
#     M = V_diff @ V_diff.T    # This is (K-1) x (K-1)
#     d = V_diff @ (theta - V[-1])  # This is (K-1,)
    
#     # Objective function: 0.5 * w^T M w - d^T w
#     def objective(w):
#         return 0.5 * w.T @ M @ w - d.T @ w
    
#     # Constraint: ||w||_1 = 1
#     def l1_constraint(w):
#         return np.sum(np.abs(w)) - 1
    
#     # Initial guess: uniform weights
#     w0 = np.ones(V.shape[0]-1) / (V.shape[0]-1)
    
#     # Run optimization
#     result = minimize(
#         objective,
#         x0=w0,
#         method='SLSQP',
#         constraints=[{'type': 'eq', 'fun': l1_constraint}],
#         bounds=[(0, None) for _ in range(len(w0))]
#     )
    
#     if not result.success:
#         print(f"Warning: Optimization did not converge. Status: {result.message}")
    
#     # Compute final distance
#     dist = np.sum((theta - V[-1])**2) + 2 * result.fun
#     return np.sqrt(max(dist, 0))

# def vertices_est(R: np.ndarray, K0: int, m: int, num_restart: int) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Find vertices of the simplex.
    
#     Parameters:
#     -----------
#     R : np.ndarray
#         Matrix of shape (p, K-1)
#     K0 : int
#         Number of potential vertices to consider
#     m : int
#         Number of kmeans centers
#     num_restart : int
#         Number of kmeans restarts
    
#     Returns:
#     --------
#     Tuple[np.ndarray, np.ndarray]
#         Selected vertices and all potential vertices
#     """
#     K = R.shape[1] + 1
    
#     # Step 2a: KMeans clustering
#     kmeans = KMeans(n_clusters=m, n_init=num_restart)
#     kmeans.fit(R)
#     theta = kmeans.cluster_centers_
#     theta_original = theta.copy()
    
#     # Step 2b': Find initial vertices
#     inner = theta @ theta.T
#     distance = np.diag(inner)[:, np.newaxis] + np.diag(inner) - 2 * inner
#     top2_idx = np.unravel_index(np.argmax(distance), distance.shape)
#     theta0 = theta[list(top2_idx)]
#     mask = np.ones(len(theta), dtype=bool)
#     mask[list(top2_idx)] = False
#     theta = theta[mask]
    
#     # Greedy selection
#     if K0 > 2:
#         for _ in range(K0 - 2):
#             if len(theta) == 0:
#                 break
#             inner = theta @ theta.T
#             distance = np.ones((len(theta0), 1)) @ np.diag(inner)[np.newaxis, :] - 2 * theta0 @ theta.T
#             ave_dist = np.mean(distance, axis=0)
#             idx = np.argmax(ave_dist)
#             theta0 = np.vstack([theta0, theta[idx]])
#             theta = np.delete(theta, idx, axis=0)
#         theta = theta0
    
#     # Find best K vertices
#     if len(theta) < K:
#         # If we don't have enough vertices, return what we have
#         return theta, theta_original
        
#     combinations = list(itertools.combinations(range(len(theta)), K))
#     max_values = np.zeros(len(combinations))
    
#     for i, comb in enumerate(combinations):
#         vertices = theta[list(comb)]
#         for j in range(len(theta)):
#             max_values[i] = max(simplex_dist(theta[j], vertices), max_values[i])
    
#     best_comb = combinations[np.argmin(max_values)]
#     selected_vertices = theta[list(best_comb)]
    
#     return selected_vertices, theta_original

# def test_topic_score():
#     # Set random seed for reproducibility
#     np.random.seed(42)
    
#     # Generate random matrix
#     p = 1000  # number of words
#     n = 200   # number of documents
#     K = 3     # number of topics
    
#     # Generate random matrix and normalize columns to sum to 1
#     X = np.random.rand(p, n)
#     X = X / X.sum(axis=0)[np.newaxis, :]
    
#     print("Matrix properties:")
#     print("Shape:", X.shape)
#     print("Column sums close to 1:", np.allclose(X.sum(axis=0), 1.0))
#     print("Contains NaN:", np.any(np.isnan(X)))
#     print("Contains Inf:", np.any(np.isinf(X)))
#     print("Min value:", X.min())
#     print("Max value:", X.max())
    
#     try:
#         print("\nRunning topic_score...")
#         result = topic_score(K, X)
        
#         A = result['A_hat']
#         print("\nResults:")
#         print("A_hat shape:", A.shape)
#         print("A_hat column sums close to 1:", np.allclose(A.sum(axis=0), 1.0))
#         print("A_hat contains NaN:", np.any(np.isnan(A)))
#         print("A_hat contains Inf:", np.any(np.isinf(A)))
#         print("A_hat min value:", A.min())
#         print("A_hat max value:", A.max())
#         print("\nSuccess!")
        
#     except Exception as e:
#         import traceback
#         print(f"\nAn error occurred: {str(e)}")
#         print("\nFull traceback:")
#         print(traceback.format_exc())

# # if __name__ == "__main__":
# #     import numpy as np
# #     from sklearn.cluster import KMeans
# #     from scipy.optimize import minimize
# #     import itertools
# #     test_topic_score()




import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import warnings
import itertools

def topic_score(K: int, X: np.ndarray, K0: int = None, m: int = None, 
                Mquantile: float = 0, num_restart: int = 1, seed: int = None) -> dict:
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
    V, theta = vertices_est(R, K0, m, num_restart)
    
    # Calculate convex combinations
    augmented_R = np.column_stack([R, np.ones(p)])
    augmented_V = np.column_stack([V, np.ones(K)])
    
    try:
        Pi = augmented_R @ np.linalg.inv(augmented_V)
    except np.linalg.LinAlgError:
        Pi = augmented_R @ np.linalg.pinv(augmented_V)
    
    # Ensure non-negative values
    Pi = np.maximum(Pi, 0)
    
    # Normalize Pi rows to sum to 1
    Pi = Pi / (Pi.sum(axis=1, keepdims=True) + epsilon)
    
    # Calculate word-topic matrix
    A_hat = np.sqrt(M_trunk)[:, np.newaxis] * Xi[:, 0:1] * Pi
    
    # Ensure columns sum to 1 (probability distributions over vocabulary)
    col_sums = A_hat.sum(axis=0)
    # If any column sums to nearly zero, replace it with uniform distribution
    for j in range(K):
        if col_sums[j] < epsilon:
            A_hat[:, j] = 1/p  # Uniform distribution over vocabulary
        else:
            A_hat[:, j] = A_hat[:, j] / col_sums[j]
    
    # Verify all columns sum to 1
    assert np.allclose(A_hat.sum(axis=0), 1.0), "Columns must sum to 1"
    assert np.all(A_hat >= 0), "All entries must be non-negative"
    
    return {
        'A_hat': A_hat,
        'R': R,
        'V': V,
        'Pi': Pi,
        'theta': theta
    }

def vertices_est(R: np.ndarray, K0: int, m: int, num_restart: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Find vertices of the simplex with faster implementation.
    
    Parameters:
    -----------
    R : np.ndarray
        Matrix of shape (p, K-1)
    K0 : int
        Number of potential vertices to consider
    m : int
        Number of kmeans centers
    num_restart : int
        Number of kmeans restarts
    
    Returns:
    --------
    tuple
        (Selected vertices, All potential vertices)
    """
    K = R.shape[1] + 1
    
    # Step 2a: KMeans clustering with fewer centers
    m = min(m, 5*K)  # Limit number of centers
    kmeans = KMeans(n_clusters=m, n_init=num_restart)
    kmeans.fit(R)
    theta = kmeans.cluster_centers_
    theta_original = theta.copy()
    
    # Step 2b': Find initial vertices
    inner = theta @ theta.T
    distance = np.diag(inner)[:, np.newaxis] + np.diag(inner) - 2 * inner
    top2_idx = np.unravel_index(np.argmax(distance), distance.shape)
    theta0 = theta[list(top2_idx)]
    mask = np.ones(len(theta), dtype=bool)
    mask[list(top2_idx)] = False
    theta = theta[mask]
    
    # Greedy selection
    if K0 > 2:
        for _ in range(K-2):  # Only select needed vertices
            if len(theta) == 0:
                break
            inner = theta @ theta.T
            distance = np.ones((len(theta0), 1)) @ np.diag(inner)[np.newaxis, :] - 2 * theta0 @ theta.T
            ave_dist = np.mean(distance, axis=0)
            idx = np.argmax(ave_dist)
            theta0 = np.vstack([theta0, theta[idx]])
            theta = np.delete(theta, idx, axis=0)
    
    return theta0[:K], theta_original

def simplex_dist_batch(points: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute L2 distances between points and simplex in batch.
    
    Parameters:
    -----------
    points : np.ndarray
        Points matrix of shape (N, K-1)
    V : np.ndarray
        Vertices matrix of shape (K, K-1)
    
    Returns:
    --------
    np.ndarray
        Array of distances
    """
    # Compute differences from last vertex
    V_diff = V[:-1] - V[-1]  # (K-1) x (K-1)
    
    # Compute distances using vectorized operations
    diffs = points - V[-1]  # (N, K-1)
    M = V_diff @ V_diff.T  # (K-1, K-1)
    
    # Solve for all points at once using least squares
    w = np.linalg.lstsq(V_diff, diffs.T, rcond=None)[0].T  # (N, K-1)
    
    # Compute distances
    dist = np.sum(diffs**2, axis=1) + np.sum(w @ M * w, axis=1)
    return np.sqrt(np.maximum(dist, 0))



def print_top_words_per_topic(A: np.ndarray, vocabulary: np.ndarray, n_words: int = 10, threshold: float = 0.01):
    """
    Print the most representative words for each topic.
    
    Parameters:
    -----------
    A : np.ndarray
        Word-topic matrix where each column is a topic distribution
    vocabulary : np.ndarray
        Array of vocabulary words
    n_words : int
        Number of top words to display per topic
    threshold : float
        Minimum probability threshold to display a word
    """
    n_topics = A.shape[1]
    
    for topic_idx in range(n_topics):
        # Get the word probabilities for this topic
        topic_dist = A[:, topic_idx]
        
        # Get indices of words sorted by probability
        top_word_indices = np.argsort(-topic_dist)
        
        # Get top words and their probabilities
        top_words = [(vocabulary[idx], topic_dist[idx]) 
                    for idx in top_word_indices 
                    if topic_dist[idx] > threshold][:n_words]
        
        print(f"\nTopic {topic_idx + 1}:")
        for word, prob in top_words:
            print(f"  {word}: {prob:.4f}")

def test_topic_score():
    """
    Test function for topic_score implementation.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random matrix
    p = 10000  # number of words
    n = 1592   # number of documents
    K = 15    # number of topics
    
    # Generate random matrix and normalize columns to sum to 1
    X = np.random.rand(p, n)
    X = X / X.sum(axis=0)[np.newaxis, :]
    
    print("Matrix properties:")
    print("Shape:", X.shape)
    print("Column sums close to 1:", np.allclose(X.sum(axis=0), 1.0))
    print("Contains NaN:", np.any(np.isnan(X)))
    print("Contains Inf:", np.any(np.isinf(X)))
    print("Min value:", X.min())
    print("Max value:", X.max())
    
    try:
        print("\nRunning topic_score...")
        result = topic_score(K, X)
        
        A = result['A_hat']
        print("\nResults:")
        print("A_hat shape:", A.shape)
        print("A_hat column sums close to 1:", np.allclose(A.sum(axis=0), 1.0))
        print("A_hat contains NaN:", np.any(np.isnan(A)))
        print("A_hat contains Inf:", np.any(np.isinf(A)))
        print("A_hat min value:", A.min())
        print("A_hat max value:", A.max())
        print("\nSuccess!")
        
    except Exception as e:
        import traceback
        print(f"\nAn error occurred: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())

# if __name__ == "__main__":
#     test_topic_score()