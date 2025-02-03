from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io
import base64



def calculate_silhouette():
    if request.method == 'POST':
        # Check if the file is part of the request
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        
        # Validate file name
        if file.filename == '':
            return "No selected file", 400
        
        # Load the CSV file
        df = pd.read_csv(file)

        # Define the features for clustering
        features = ['total_contributors', 'total_open_issues', 'total_closed_issues',
                    'total_stars', 'total_forks', 'total_commits', 'days_since_last_commit']
        
        
        
        # Extract the data for clustering
        X = df[features]

        # Calculate Silhouette Scores for k=2 to k=10
        silhouette_scores = []
        k_values = range(2, 11)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=300)
            kmeans.fit(X)
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)

        # Create a plot for Silhouette Scores
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='blue', label='Silhouette Score')
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Scores for K-Means Clustering', fontsize=14)
        plt.xticks(k_values)
        plt.legend(loc='best')
        plt.tight_layout()
        

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        print("\nSilhouette Scores:")
        print(f"{'k':<5} {'Silhouette Score':<20}")
        print("-" * 25)
        for k, score in zip(k_values, silhouette_scores):
            print(f"{k:<5} {score:<20.4f}")

        # Encode plot to base64 for HTML embedding
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Display silhouette scores below the plot
        return render_template('result/silhouetteScore.html', 
                               plot_url=plot_url, 
                               silhouette_scores=silhouette_scores,
                               k_values=k_values,
                               zip=zip)
