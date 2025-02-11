import pandas as pd
import glob
import numpy as np

# Read and concatenate all CSV files
files = glob.glob('results/*_predictions.csv')
df_list = [pd.read_csv(f) for f in files]
combined_df = pd.concat(df_list, ignore_index=True)

# Convert all labels to lowercase for consistent comparison
combined_df['true_label'] = combined_df['true_label'].str.lower()
method_columns = [col for col in combined_df.columns if '_pred' in col]
for col in method_columns:
    combined_df[col] = combined_df[col].str.lower()

# Initialize dictionary to store errors per fold
fold_errors = {method: [] for method in method_columns}

# Calculate error for each method per fold
for fold_num in range(5):
    fold_df = combined_df[combined_df['fold'] == fold_num]
    for method in method_columns:
        error = 1 - (fold_df['true_label'] == fold_df[method]).mean()
        fold_errors[method].append(error)

# Create results DataFrame
results = pd.DataFrame({
    'method': method_columns,
    'mean_error': [np.mean(fold_errors[method]) for method in method_columns],
    'variance_error': [np.var(fold_errors[method], ddof=1) for method in method_columns]
})

# Sort results by method name
results = results.sort_values('method').reset_index(drop=True)

# Display results
print("Error metrics across folds:")
print(results)

# # Optional: Save to CSV
# results.to_csv('results/method_errors_summary.csv', index=False)