from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
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
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        # Get centroids
        centroids = kmeans.cluster_centers_

        # Calculate Silhouette Score
        silhouette_avg = silhouette_score(X, kmeans.labels_)

        # Create the plot
        cluster_colors = plt.cm.get_cmap('tab10', n_clusters).colors  # Generate colors for n_clusters
        plt.figure(figsize=(10, 6))

        for cluster_idx in range(n_clusters):
            cluster_points = X[kmeans.labels_ == cluster_idx]
            plt.scatter(cluster_points.iloc[:, 3], cluster_points.iloc[:, 4],  # Use 'total_stars' and 'total_forks' for visualization
                        c=[cluster_colors[cluster_idx]], 
                        alpha=0.6, 
                        label=f'Cluster {cluster_idx}')

        # Plot centroids with red 'X' marker
        plt.scatter(centroids[:, 3], centroids[:, 4], c='red', marker='X', s=300, label='Centroids')

        # Customize the plot
        plt.xlabel(features[3], fontsize=12)  # 'total_stars'
        plt.ylabel(features[4], fontsize=12)  # 'total_forks'
        plt.title(f'K-Means Clusters (k={n_clusters})', fontsize=14)
        plt.legend(loc='best')

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()

        # Encode plot to base64 for HTML embedding
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')

        centroid_values = []
        for i, centroid in enumerate(centroids):
        centroid_info = {features[j]: f'{centroid[j]:.4f}' for j in range(len(features))}
        centroid_values.append(centroid_info)

        return render_template('result/analisisKMeans.html',
                               plot_url=plot_url,
                               silhouette_score=silhouette_avg,
                               centroids=centroids,
                               n_clusters=n_clusters)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
