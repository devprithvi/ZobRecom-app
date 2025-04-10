import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import kagglehub

# Download dataset using KaggleHub
path = kagglehub.dataset_download("promptcloud/jobs-on-naukricom")

print("Path to dataset files:", path)

# Load dataset directly from the path provided by KaggleHub
file_path = path + '/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv'
data = pd.read_csv(file_path)

# Show the first few rows of the dataset
data.head()

# Preprocessing
data['Job Title'] = data['Job Title'].fillna('')
data['Key Skills'] = data['Key Skills'].fillna('')
data['combined'] = data['Job Title'] + " " + data['Key Skills']

# Function to extract minimum years of experience from a range like '0 - 5 yrs'
def extract_min_experience(experience_range):
    try:
        # Extract the numeric value before the dash (e.g., "0" in "0 - 5 yrs")
        return int(experience_range.split(' - ')[0])
    except:
        return 0  # If there's an issue with the format, return 0 years

# Apply the extraction function to the 'Job Experience Required' column
data['min_experience'] = data['Job Experience Required'].apply(extract_min_experience)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined'])

# Recommendation function
def recommend_jobs(user_input, min_experience, top_n=5):
    # Filter based on minimum experience
    filtered_data = data[data['min_experience'] >= min_experience]
    
    # Check if the filtered data is empty
    if filtered_data.empty:
        return None, len(filtered_data)  # Return empty recommendations and the count of matching jobs
    
    # Vectorize user input
    user_vec = tfidf_vectorizer.transform([user_input])
    cosine_sim_user = cosine_similarity(user_vec, tfidf_matrix[filtered_data.index])
    
    # Get the most similar jobs
    similar_jobs = cosine_sim_user.argsort()[0, -top_n:][::-1]
    return filtered_data.iloc[similar_jobs], len(filtered_data)

# Streamlit app interface
st.title('Job Recommendation System')

# Get user input for skills/job title
user_input = st.text_input("Enter your job title or skills:")

# Get user input for minimum years of experience
min_experience = st.slider("Select Minimum Years of Experience", min_value=0, max_value=30, value=0, step=1)

# Display recommendations if the user input is provided
if user_input:
    recommendations, filtered_count = recommend_jobs(user_input, min_experience)
    
    if recommendations is None:
        st.write(f"No jobs found with at least {min_experience} years of experience. Try reducing the experience requirement.")
    else:
        st.write(f'Top Recommended Jobs for {user_input} with at least {min_experience} years of experience:')
        st.dataframe(recommendations[['Job Title', 'Location', 'Job Salary', 'Industry', 'Job Experience Required']])
    
    # Provide additional feedback based on the number of jobs found
    if filtered_count == 0:
        st.write("No jobs meet the current experience filter. Please adjust your filter to get more results.")
