from flask import request, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import io
import base64


def analisis_kmeans():
    # Check if file is uploaded
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    # Get number of clusters from form
    n_clusters = int(request.form.get('n_clusters', 2))  # Default to 2 if not specified

    # Load dataset
    df = pd.read_csv(file)
    features = ['total_contributors', 'total_open_issues', 'total_closed_issues',
                'total_stars', 'total_forks', 'total_commits', 'days_since_last_commit']
    X = df[features]
    
    # Apply K-Means with user-specified k
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=300)
    kmeans.fit(X)

    # Get centroids
    centroids = kmeans.cluster_centers_

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(X, kmeans.labels_)

    # Create the plot
    cluster_colors = plt.cm.get_cmap('tab10', n_clusters).colors  # Generate colors for n_clusters
    plt.figure(figsize=(10, 6))

    # Create visualization
    for cluster_idx in range(n_clusters):
        cluster_points = X[kmeans.labels_ == cluster_idx]
        plt.scatter(
            cluster_points['total_stars'],  # Using 'total_stars' for x-axis
            cluster_points['total_forks'],  # Using 'total_forks' for y-axis
            c=[cluster_colors[cluster_idx]], 
            alpha=0.6, 
            label=f'Cluster {cluster_idx}'
        )

    # Plot centroids with red X markers
    plt.scatter(
        centroids[:, 3], centroids[:, 4],  # Use 'total_stars' and 'total_forks' for centroids
        c='red',
        marker='X',
        s=250,
        label='Centroids',
        edgecolor='black'
    )

    # Customize plot
    plt.xlabel(features[3], fontsize=12)  # 'total_stars'
    plt.ylabel(features[4], fontsize=12)  # 'total_forks'
    plt.title(f'K-Means Clusters (k={n_clusters})', fontsize=14)
    plt.legend(loc='best')

    # Remove grid
    plt.grid(False)

    # Prepare the centroid values with feature names
    centroid_values = []
    for i, centroid in enumerate(centroids):
        centroid_info = {features[j]: f'{centroid[j]:.4f}' for j in range(len(features))}
        centroid_values.append(centroid_info)

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()

    # Encode plot to base64 for HTML embedding
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')

# Pass this `centroid_values` to the template as it is now a list of dictionaries
    return render_template('result/analisisKMeans.html', 
                       plot_url=plot_url, 
                       silhouette_score=silhouette_avg, 
                       centroid_values=centroid_values, 
                       n_clusters=n_clusters)   