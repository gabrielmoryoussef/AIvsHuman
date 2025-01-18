import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('final_author_pair_gpt_results_with_rf.csv')

# # Filter out excluded authors
# exclude_authors = ['Piotr Kokoszka', 'Sanat K. Sarkar']
# df = df[~df['author1'].isin(exclude_authors) & ~df['author2'].isin(exclude_authors)]

# Set figure style
plt.figure(figsize=(6, 4))  # Increased figure width for better label spacing

# Define methods and their labels
methods = {
    'Random Forest + ST_accuracy': 'BERT',  # Split into two lines
    'Topic SCORE_accuracy': 'T-SCORE',
    'PHC_accuracy': 'HC',
    'LLM_PHC_accuracy': 'HC-GPT',
    'LLM_Pure_accuracy': 'GPT'
}

# Prepare data for plotting
plot_data = []
labels = []
for col, label in methods.items():
    plot_data.append(1 - df[col].values)
    labels.append(label)

# Create box plot
box_props = dict(linestyle='-', linewidth=2)
whisker_props = dict(linestyle='-', linewidth=1.5)
flier_props = dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5)
median_props = dict(linestyle='-', linewidth=2.5, color='black')

plt.boxplot(plot_data,
            patch_artist=True,
            labels=labels,
            boxprops=box_props,
            whiskerprops=whisker_props,
            flierprops=flier_props,
            medianprops=median_props)

# Customize colors
colors = ['#3182bd', '#31a354', '#756bb1', '#636363', '#de2d26']  # Added one more color
boxes = plt.gca().patches
for box, color in zip(boxes, colors):
    box.set_facecolor(color)
    box.set_alpha(0.7)

# Customize plot
plt.ylabel('Error Rate', fontsize=14)  # Increased y-label size
plt.grid(True, linestyle='--', alpha=0.3)
plt.ylim(0, 1)

# Add reference line at 0.5
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)

# Rotate x-axis labels and make them bigger
plt.xticks(rotation=45, ha='right', fontsize=12)  # Increased font size to 12

# Add more space at the bottom for labels
plt.subplots_adjust(bottom=0.2)

# Save the plot
plt.savefig('method_error_comparison_boxplots_gpt.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\nSummary Statistics (Error Rates):")
for col, label in methods.items():
    print(f"\n{label}:")
    error_stats = (1 - df[col]).describe()
    print(f"Mean Error: {error_stats['mean']:.3f}")
    print(f"Std Error: {error_stats['std']:.3f}")
    print(f"Min Error: {error_stats['min']:.3f}")
    print(f"25% Error: {error_stats['25%']:.3f}")
    print(f"Median Error: {error_stats['50%']:.3f}")
    print(f"75% Error: {error_stats['75%']:.3f}")
    print(f"Max Error: {error_stats['max']:.3f}")
    print(f"Number of samples: {len(df[col])}")