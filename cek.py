import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset/dataset_update.csv')
features = ['total_contributors', 'total_open_issues', 'total_closed_issues',
            'total_stars', 'total_forks', 'total_commits', 'days_since_last_commit']
X = df[features]

# Calculate Silhouette Scores for k=2 to k=10
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

# Visualize Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='blue', label='Silhouette Score')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Scores for K-Means Clustering', fontsize=14)
plt.xticks(k_values)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Display silhouette scores below the plot in a table format
print("\nSilhouette Scores:")
print(f"{'k':<5} {'Silhouette Score':<20}")
print("-" * 25)
for k, score in zip(k_values, silhouette_scores):
    print(f"{k:<5} {score:<20.4f}")