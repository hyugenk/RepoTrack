from flask import Flask, request, render_template, jsonify, send_file
from sklearn.preprocessing import MinMaxScaler
import requests
import pandas as pd
import csv
import time
from datetime import datetime

from KMeans import *
from SilhouetteScore import *


app = Flask(__name__)

def get_all_commits(owner, repo, username, token):
    commits = []
    page = 1
    try:
        while True:
            url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            headers = {"Authorization": f"token {token}"}
            params = {
                "author": username,
                "per_page": 100,
                "page": page
            }
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                page_commits = response.json()
                if not page_commits:
                    break
                commits.extend(page_commits)
                page += 1
                # Menambahkan delay untuk menghindari rate limiting
                time.sleep(0.1)
            elif response.status_code == 403:
                print(f"Rate limit exceeded for {username}")
                time.sleep(60)  # Tunggu 1 menit jika rate limit tercapai
                continue
            else:
                print(f"Error fetching commits for {username}: {response.status_code}")
                break
    except Exception as e:
        print(f"Exception while fetching commits for {username}: {str(e)}")
    return commits

def get_commit_dates(owner, repo, username, token):
    try:
        commits = get_all_commits(owner, repo, username, token)
        if commits:
            first_commit = min(commits, key=lambda c: c['commit']['author']['date'])
            last_commit = max(commits, key=lambda c: c['commit']['author']['date'])
            return (
                first_commit['commit']['author']['date'],
                last_commit['commit']['author']['date']
            )
    except Exception as e:
        print(f"Error getting commit dates for {username}: {str(e)}")
    return None, None

def get_all_contributors(owner, repo, token):
    contributors = []
    page = 1
    try:
        while True:
            url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
            headers = {"Authorization": f"token {token}"}
            params = {"per_page": 100, "page": page}
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                page_contributors = response.json()
                if not page_contributors:
                    break
                contributors.extend(page_contributors)
                page += 1
                time.sleep(0.1)  # Delay untuk menghindari rate limiting
            elif response.status_code == 403:
                print("Rate limit exceeded, waiting...")
                time.sleep(60)
                continue
            else:
                print(f"Error fetching contributors: {response.status_code}")
                break
    except Exception as e:
        print(f"Exception while fetching contributors: {str(e)}")
    return contributors

def get_user_details(username, token):
    try:
        url = f"https://api.github.com/users/{username}"
        headers = {"Authorization": f"token {token}"}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            user_data = response.json()
            return user_data.get("created_at", None)
        elif response.status_code == 403:
            print(f"Rate limit exceeded for user {username}")
            time.sleep(60)
            return get_user_details(username, token)  # Retry after waiting
    except Exception as e:
        print(f"Error getting user details for {username}: {str(e)}")
    return None

def get_repo_owner(repo_name, token):
    search_url = f"https://api.github.com/search/repositories?q={repo_name}"
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        return None

    search_results = response.json()
    if search_results.get('total_count', 0) == 0:
        return None

    return search_results['items'][0]['owner']['login']

def contributor_analyzer(contributor, owner, repo, token):
    username = contributor['login']
    contributions = contributor['contributions']
    
    retry_count = 0
    while retry_count < 3:
        first_commit_date, last_commit_date = get_commit_dates(owner, repo, username, token)
        if first_commit_date and last_commit_date:
            break
        retry_count += 1
        time.sleep(2)
    
    retry_count = 0
    created_at = None
    while retry_count < 3:
        created_at = get_user_details(username, token)
        if created_at:
            break
        retry_count += 1
        time.sleep(2)
    
    if first_commit_date and last_commit_date:
        first_date = datetime.strptime(first_commit_date, "%Y-%m-%dT%H:%M:%SZ")
        last_date = datetime.strptime(last_commit_date, "%Y-%m-%dT%H:%M:%SZ")
        days_between = (last_date - first_date).days
    else:
        days_between = None
    
    results = {
        "owner": owner,
        "repo_name": repo,
        "username": username,
        "account_created_at": created_at,
        "total_contributors": contributions,
        "first_commit_date": first_commit_date,
        "last_commit_date": last_commit_date,
        "days_between_commits": days_between
    }
    
    return results

def get_repo_info(owner, repo_name, token):
    # URL API GitHub
    base_url = f"https://api.github.com/repos/{owner}/{repo_name}"
    commits_url = f"{base_url}/commits"

    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # Ambil informasi dasar repositori
    response_repo = requests.get(base_url, headers=headers)
    if response_repo.status_code != 200:
        return None
    
    repo_data = response_repo.json()
    total_stars = repo_data.get('stargazers_count', 0)
    total_forks = repo_data.get('forks_count', 0)

    # Ambil jumlah contributors
    contributors_url = f"{base_url}/contributors"
    response_contributors = requests.get(contributors_url, headers=headers, params={'per_page': 1, 'anon': 1})
    if response_contributors.status_code != 200:
        return None
        
    contributors = response_contributors.links.get('last', {}).get('url', None)
    total_contributors = 1 if not contributors else int(contributors.split('page=')[-1])

    # Ambil jumlah issues (open dan closed)
    issues_url = f"{base_url}/issues"

    # Open issues
    response_open_issues = requests.get(issues_url, headers=headers, params={'state': 'open', 'per_page': 1})
    if response_open_issues.status_code != 200:
        return None
        
    open_issues = response_open_issues.links.get('last', {}).get('url', None)
    total_open_issues = 1 if not open_issues else int(open_issues.split('page=')[-1])

    # Closed issues
    response_closed_issues = requests.get(issues_url, headers=headers, params={'state': 'closed', 'per_page': 1})
    if response_closed_issues.status_code != 200:
        return None
        
    closed_issues = response_closed_issues.links.get('last', {}).get('url', None)
    total_closed_issues = 1 if not closed_issues else int(closed_issues.split('page=')[-1])

    # Ambil total commits
    params_total_commits = {'per_page': 1}
    response_total_commits = requests.get(commits_url, headers=headers, params=params_total_commits)
    if response_total_commits.status_code != 200:
        return None

    total_commits = 1
    if 'last' in response_total_commits.links:
        total_commits = int(response_total_commits.links['last']['url'].split('page=')[-1])

    # Ambil informasi last commit
    response_last_commit = requests.get(commits_url, headers=headers, params={'per_page': 1})
    if response_last_commit.status_code != 200:
        return None
        
    last_commit_data = response_last_commit.json()[0]
    last_commit_date = last_commit_data['commit']['committer']['date']
    last_commit_date = datetime.strptime(last_commit_date, "%Y-%m-%dT%H:%M:%SZ")
    current_date = datetime.now()

    days_since_last_commit = (current_date - last_commit_date).days
    if days_since_last_commit < 0:
        days_since_last_commit = 0

    return {
        'repo_name': repo_name,
        'owner': owner,
        'total_contributors': total_contributors,
        'total_stars': total_stars,
        'total_forks': total_forks,
        'total_open_issues': total_open_issues,
        'total_closed_issues': total_closed_issues,
        'total_commits': total_commits,
        'days_since_last_commit': days_since_last_commit
    }

@app.route('/')
def index():
    return render_template('index.html')

# FINAL
@app.route('/start', methods=['POST'])
def start_analysis():
    owner = request.form['owner']
    repository = request.form['repository']
    token = request.form['github_token']

    results = [] 
    results.append(get_repo_info(owner, repository, token))
    return render_template('result/repositories.html', results=results)

# FINAL
@app.route('/analyze-dataset', methods=['POST'])
def analyze_dataset():
    token = request.form['github_token']
    file = request.files['dataset']

    if not file:
        return "No file uploaded", 400

    # Read the uploaded CSV file
    dataset = pd.read_csv(file)
    results = []

    for _, row in dataset.iterrows():
        repo_name = row['repo_name']
        # Jika owner tidak ada di CSV, cari menggunakan API
        owner = row.get('owner') or get_repo_owner(repo_name, token)
        
        if owner:
            repo_info = get_repo_info(owner, repo_name, token)
            if repo_info:
                results.append(repo_info)

    # Save results to a new CSV file
    output_file = 'analyzed_dataset.csv'
    if results:
        keys = results[0].keys()
        with open(output_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

    return render_template('result/repositories.html', results=results, download_link=output_file if results else None)

@app.route('/analyze-contributor-dataset', methods=['POST'])
def analyze_contributor_dataset():
    token = request.form['github_token']
    file = request.files['dataset']

    if not file:
        return "No file uploaded", 400

    # Read the uploaded CSV file
    dataset = pd.read_csv(file)
    results = []
    
    for _, row in dataset.iterrows():
        owner = row['owner']
        repo = row['repo_name']
        contributors = get_all_contributors(owner, repo, token)
        for contributor in contributors:
            results.append(contributor_analyzer(contributor, owner, repo, token))

    # Save results to a new CSV file
    output_file = 'analyzed_contributor_dataset.csv'
    if results:
        keys = results[0].keys()
        with open(output_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

    return render_template('result/contributors.html', results=results, download_link=output_file if results else None)

# FINAL
# Contributor Analysis
@app.route('/analyze-contributors', methods=['POST'])
def analyze_contributors():
    owner = request.form['owner']
    repo = request.form['repository']
    token = request.form['github_token']
    results = []
    
    try:
        contributors = get_all_contributors(owner, repo, token)
        
        for contributor in contributors:
            try:
                res = contributor_analyzer(contributor, owner, repo, token)
                time.sleep(0.1)
                if res:
                    results.append(res)
                
            except Exception as e:
                print(f"Error processing contributor {username}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in analyze_contributors: {str(e)}")
    return render_template('result/contributors.html', results=results)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)


# Menghapus duplikasi route download_file dan menggantinya dengan satu implementasi
# @app.route('/download/<filename>')
# def download_file(filename):
#     try:
#         return send_file(filename, as_attachment=True)
#     except Exception as e:
#         return str(e), 404

# ... (route lainnya tetap sama)


def preprocess_dataset(file, options):
    try:
        # Baca file CSV
        df = pd.read_csv(file)
        
        # Simpan informasi preprocessing
        preprocessing_info = {
            "original_shape": df.shape,
            "steps_taken": []
        }
        
        # Hapus duplikat jika dipilih
        if options.get('remove_duplicates'):
            initial_rows = len(df)
            df = df.drop_duplicates()
            preprocessing_info["steps_taken"].append({
                "step": "remove_duplicates",
                "rows_removed": initial_rows - len(df)
            })
        
        # Format tanggal jika dipilih
        if options.get('format_dates'):
            
            selected_date_columns = request.form.getlist('date_columns')
            formatted_columns = []
            not_found_columns = []
            for col in selected_date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                        # print(df[col])
                        formatted_columns.append(col)
                    except Exception as e:
                        preprocessing_info["steps_taken"].append({
                            "step": "format_dates",
                            "error": f"Failed to format column {col}: {str(e)}"
                        })
                else:
                    print("NOT FOUND")
                    not_found_columns.append(col)
            
            # Menambahkan informasi preprocessing untuk format tanggal
            if formatted_columns:
                preprocessing_info["steps_taken"].append({
                    "step": "format_dates",
                    "columns_formatted": formatted_columns,
                    "format": "YYYY-MM-DD"
                })
            
            if not_found_columns:
                preprocessing_info["steps_taken"].append({
                    "step": "missing_columns",
                    "columns": not_found_columns,
                    "message": "These date columns were not found in the dataset"
                })
        
    
        
        preprocessing_info["final_shape"] = df.shape
        
        return df, preprocessing_info
        
    except Exception as e:
        return None, {"error": str(e)}

@app.route('/preprocess-dataset', methods=['POST'])
def handle_preprocessing():
    if 'dataset' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    options = {
        'remove_duplicates': request.form.get('remove_duplicates') == 'on',
        'format_dates': request.form.get('format_dates') == 'on',
        # 'convert_numeric': request.form.get('convert_numeric') == 'on'
    }
    print(f"x::{options}")
    processed_df, preprocessing_info = preprocess_dataset(file, options)
    
    if processed_df is None:
        return render_template('result/processing.html', 
                             error=preprocessing_info["error"])
    
    # Simpan hasil preprocessing
    output_filename = 'preprocessed_' + file.filename
    processed_df.to_csv(output_filename, index=False)
    
    return render_template('result/processing.html',
                          preprocessing_info=preprocessing_info,
                          output_file=output_filename)

@app.route('/download/<filename>')
def download_csv(filename):
    try:
        filepath = f"./downloads/{filename}"
        return send_file(
            filename,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return str(e)

@app.route('/silhouette', methods=['POST'])
def silhouette_route():
    return calculate_silhouette()

@app.route('/analisis', methods=['POST'])
def kmeans_route():
    return analisis_kmeans()

if __name__ == '__main__':
    app.run(debug=True)




