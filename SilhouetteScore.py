from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io
import base64

def calculate_silhouette():
    # Validate file upload
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    # Validate file name
    if file.filename == '':
        return "No selected file", 400
    
    # Read the CSV file
    data = pd.read_csv(file)
    
    # Ensure numeric columns exist
    numeric_columns = data.select_dtypes(include=[np.number])
    
    if numeric_columns.empty:
        return "No numeric columns found in the dataset", 400
    
    # Check if the dataset has enough samples for clustering
    n_samples = numeric_columns.shape[0]
    if n_samples < 2:
        return "Not enough samples for clustering (at least 2 required)", 400
    
    # Preprocessing: Normalize the numerical data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numeric_columns)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for clustering
    data_pca = pca.fit_transform(data_scaled)
    
    # Calculate Silhouette Scores for clusters ranging from 2 to 10
    silhouette_scores = []
    clusters_range = list(range(2, min(n_samples, 10) + 1))  # Ensure n_clusters <= n_samples
    
    for n_clusters in clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data_pca)
        silhouette_avg = silhouette_score(data_pca, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(clusters_range, silhouette_scores, marker='o', linestyle='-', color='blue')
    plt.title('Silhouette Scores for K-Means Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(clusters_range)
    plt.grid()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode plot to base64 for HTML embedding
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return render_template('result/silhouetteScore.html', 
                           plot_url=plot_url, 
                           silhouette_scores=silhouette_scores,
                           clusters_range=clusters_range,
                           zip=zip)