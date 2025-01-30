# RepoTrack

**RepoTrack** is a web-based application designed to **track, analyze, and visualize data from GitHub repositories** efficiently. Built using the Flask framework, RepoTrack provides an intuitive interface for users to explore key information about GitHub projects, including project activity, developer contributions, and commit trends.

## Features

### 1. **Project Data Collection**
- Fetch repository data from GitHub by specifying the owner and repository name.
- Supports batch data collection through CSV file uploads.

### 2. **Contributor Analysis**
- Analyze user contributions based on the number of commits, last activity date, and engagement patterns.

### 3. **KMeans Clustering**
- Perform machine learning-based analysis to cluster projects or contributors based on specific characteristics.
- Visualize clustering results using graphs or charts for better understanding.

### 4. **Dataset Preprocessing**
- Clean data by removing duplicates, standardizing date formats, or normalizing data.

### 5. **GitHub API Integration**
- Secure data retrieval using GitHub API tokens.
- Adheres to GitHub's rate limits and ensures smooth operation.

### 6. **User-Friendly Interface**
- Responsive design using Bootstrap for professional and intuitive user interaction.
- Provides forms and buttons for tasks like data collection, contributor analysis, and dataset preprocessing.

## Who Should Use RepoTrack?

RepoTrack is ideal for:
- **Open Source Researchers** seeking to analyze project activity on GitHub.
- **GitHub Repository Owners** who want insights into contribution patterns and repository activity.
- **Students and Academics** conducting case studies on open source software.
- **Data Scientists** looking for tools to explore GitHub-based datasets.

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3 (Bootstrap), JavaScript
- **Machine Learning**: Scikit-learn (for clustering analysis)
- **Data Visualization**: Matplotlib and Chart.js

## Benefits of RepoTrack

- Provides deep insights into the activity and potential "abandonment" of GitHub projects.
- Enhances efficiency in analyzing datasets with automated preprocessing features.
- Supports data-driven decision-making with visualized analysis results.

---

## How to Get Started

1. Clone the repository:
   ```bash
   git clone https://github.com/hyugenk/RepoTrack.git

2. Navigate to the project directory:
   ```bash
   cd RepoTrack

4. (Optional) Create a virtual environment to manage dependencies:
   ```bash
  `python -m venv venv`
  `source venv/bin/activate   # On Linux/Mac`
  `venv\Scripts\activate      # On Windows`

6. Install the required dependencies:
   ```bash
   `pip install -1 requirements.txt`

8. Run the Flask application:
   ```bash
   `pyhton app.py`

10. ENJOY!
