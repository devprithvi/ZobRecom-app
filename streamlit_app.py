import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import kagglehub
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Job Recommendation System", layout="wide")


# Download dataset using KaggleHub
path = kagglehub.dataset_download("promptcloud/jobs-on-naukricom")

print("Path to dataset files:", path)

# Load dataset directly from the path provided by KaggleHub
file_path = path + '/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv'
data = pd.read_csv(file_path)

# Show the first few rows of the dataset
data.head()

data.drop(columns=['Uniq Id'], inplace=True)

# Filter job titles with counts >= 10
job_counts = data['Job Title'].value_counts()
filtered_job_titles = job_counts[job_counts >= 10].index

# Keep only the rows with job titles in the filtered list above
data = data[data['Job Title'].isin(filtered_job_titles)]

# Convert 'Crawl Timestamp' to datetime
data['Crawl Timestamp'] = pd.to_datetime(data['Crawl Timestamp'], errors='coerce')

# Extract the 'Crawl Month' from the 'Crawl Timestamp'
data['Crawl Month'] = data['Crawl Timestamp'].dt.month

data.drop(columns=['Crawl Timestamp'], inplace=True)

# Clean the data by removing rows with null values in 'Role Category' or 'Key Skills'
data_cleaned = data[~(data['Role Category'].isnull() | data['Key Skills'].isnull())].copy()

# Define the function to extract experience
def extract_experience(exp):
    exp = str(exp).lower().strip()  # Convert to lowercase and strip whitespace
    match = re.findall(r'\d+', exp)  # Find all numbers in the string

    if len(match) == 0:
        return None  # No experience found
    elif len(match) == 1:
        return int(match[0])  # Only one experience value found
    else:
        return int(match[1])  # Return the second value in case of a range

# Apply the function to extract experience
data_cleaned['Experience(Years)'] = data_cleaned['Job Experience Required'].apply(extract_experience)

# Drop the 'Job Experience Required' column as it's no longer needed
data_cleaned.drop(columns=['Job Experience Required'], inplace=True)


# Define the function to extract average salary
def extract_avg_salary(salary_str):
    salary_str = str(salary_str).lower()

    # Updating non-numeric values to NAN
    if any(x in salary_str for x in ['not disclosed', 'negotiable', 'best', 'incentive', 'as per', 'variable', 'bonus']):
        return np.nan

    # Extract all numbers (support commas, optional INR)
    salary_nums = re.findall(r'[\d,]+', salary_str)
    if len(salary_nums) >= 2:
        try:
            low = int(salary_nums[0].replace(',', ''))
            high = int(salary_nums[1].replace(',', ''))
            return (low + high) / 2  # Average
        except:
            return np.nan
    elif len(salary_nums) == 1:
        try:
            return int(salary_nums[0].replace(',', ''))
        except:
            return np.nan
    else:
        return np.nan

# Apply the function to extract average salary
data_cleaned['Avg Salary(INR)'] = data_cleaned['Job Salary'].apply(extract_avg_salary)

# Group by 'Job Title' and compute the median salary for each job title
median_by_title = data_cleaned.groupby('Job Title')['Avg Salary(INR)'].median()

# Fill missing values in 'Avg Salary(INR)' with the job title-level median
data_cleaned['Avg Salary(INR)'] = data_cleaned.apply(
    lambda row: median_by_title[row['Job Title']] if pd.isna(row['Avg Salary(INR)']) and row['Job Title'] in median_by_title else row['Avg Salary(INR)'],
    axis=1
)

# Drop 'Functional Area' and 'Industry' columns as they are no longer needed
data_cleaned.drop(columns=['Functional Area', 'Industry'], inplace=True)

# Step 3: Define mapping of target titles to reference roles
title_impute_map = {
    "Branch Service Partner": "Back Office Executive",
    "Business Development": "Business Development Executive",
    "Client Relationship Partner": "Relationship Manager",
    "UI Developer": "Developer",
    "Developer": "Developer",
    "iOS Developer": "iOS Developer"
}

# Step 4: Compute median salaries for the source roles
reference_medians = {}
for target, source in title_impute_map.items():
    mask = data_cleaned['Job Title'].str.contains(source.lower(), case=False, na=False)
    median_salary = data_cleaned.loc[mask, 'Avg Salary(INR)'].median()
    reference_medians[target] = median_salary

# Step 5: Impute missing salaries for specified target roles
def custom_impute(row):
    if pd.isna(row['Avg Salary(INR)']):
        for target_title, avg_salary in reference_medians.items():
            if target_title.lower() in row['Job Title'].lower():
                return avg_salary
    return row['Avg Salary(INR)']

data_cleaned['Avg Salary(INR)'] = data_cleaned.apply(custom_impute, axis=1)

# Step 6: Fill remaining NaNs with overall median salary
overall_median = data_cleaned['Avg Salary(INR)'].median()
data_cleaned['Avg Salary(INR)'] = data_cleaned['Avg Salary(INR)'].fillna(overall_median)

# Drop the 'Job Salary' column as it's no longer needed
data_cleaned.drop(columns=['Job Salary'], inplace=True)

# Function to get jobs that match the input title
def get_matching_jobs(title_input):
    title_clean = re.sub(r'[^a-z\s]', '', title_input.lower())  # Clean input
    return data_cleaned[data_cleaned['Job Title'].str.lower().str.contains(title_clean)]  # Filter matching jobs

# Function to calculate the job similarity score
def score_job(target_row, ref_row, weight_salary=0.3, weight_exp=0.4, weight_location=0.3):
    # Salary similarity: inverse of absolute difference
    salary_diff = abs(target_row['Avg Salary(INR)'] - ref_row['Avg Salary(INR)'])  # bigger difference = worse match

    # Experience similarity
    exp_diff = abs(target_row['Experience(Years)'] - ref_row['Experience(Years)'])  # bigger difference = worse match

    # Location match (binary score)
    location_score = 1 if target_row['Location'].lower() == ref_row['Location'].lower() else 0

    # Combine with weights
    return (
        -weight_salary * salary_diff
        -weight_exp * exp_diff
        +weight_location * location_score
    )

# Function to recommend jobs based on input
def recommend_prioritized(title_input, top_n=10):
    # Get matching jobs based on title input
    matches = get_matching_jobs(title_input)
    if matches.empty:
        return "No jobs found matching title."

    # Take the first match as reference
    ref_job = matches.iloc[0]

    # Score all other matches
    matches = matches[matches.index != ref_job.name]  # Exclude the reference job itself
    matches['match_score'] = matches.apply(lambda row: score_job(ref_job, row), axis=1)

    # Return top N based on score
    return matches.sort_values(by='match_score', ascending=False).head(top_n)[[
        'Job Title', 'Key Skills', 'Location', 'Experience(Years)', 'Avg Salary(INR)', 'match_score'
    ]]

# Streamlit user interface to input job title and get recommendations
st.title("Welcome to the Job Recommendation System")
# Show instructions and site purpose
st.write("""
    Find job recommendations based on your job title, experience, location, and salary preferences. 
    This is your gateway to career opportunities!
""")
title_input = st.text_input("Enter your Job Title to Search")

if title_input:
    top_n = 10  # Number of top recommendations to show
    recommendations = recommend_prioritized(title_input, top_n)
    st.write(recommendations)



# Optional: Add footer or additional UI features
st.markdown("""
    <footer style="text-align: center; padding-top: 20px;">
        <p>Powered by Streamlit - A Job Recommendation System</p>
    </footer>
""", unsafe_allow_html=True)