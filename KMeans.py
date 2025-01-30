# KMeans.py
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

def analisis_kmeans():
    # Check if file is uploaded
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    # Get number of clusters from form
    n_clusters = int(request.form.get('n_clusters', 2))  # Default to 2 if not specified
    
    # Read the CSV file
    data = pd.read_csv(file)
    
    # Preprocessing: Normalize the numerical data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]))
    
    # Apply PCA for dimensionality reduction to 2 components
    pca = PCA(n_components=2, random_state=42)
    data_pca = pca.fit_transform(data_scaled)
    
    # Perform K-Means clustering with user-specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_pca)
    
    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(data_pca, cluster_labels)
    
    # Extract centroids from scikit-learn KMeans
    centroids = kmeans.cluster_centers_
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        data_pca[:, 0],
        data_pca[:, 1],
        c=cluster_labels,
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    # Plot centroids
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c='red',
        marker='X',
        s=200,
        label='Centroids'
    )
    
    # Customize plot
    plt.title(f"K-Means Clusters (Silhouette Score: {silhouette_avg:.4f})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode plot to base64 for HTML embedding
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return render_template('result/analisisKMeans.html', 
                         plot_url=plot_url, 
                         silhouette_score=silhouette_avg, 
                         centroids=centroids,
                         n_clusters=n_clusters)