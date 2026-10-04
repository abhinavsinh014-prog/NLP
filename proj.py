import pandas as pd
import numpy as np
import nltk as nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# import tensorflow as tf

import pymupdf

def extract_text_from_pdf(pdf_path):
    text = ""

    pdf = pymupdf.open(pdf_path)

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


text = extract_text_from_pdf("New blank-Converted.pdf")


set1 = set(text.split())

skills = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "git",
    "docker",
    "aws",
    "streamlit",
    "opencv",
    "word2vec"
]

def extract_skills(text, skills):

    text = text.lower()

    found_skills = []

    for skill in skills:
        if skill in text:
            found_skills.append(skill)

    return found_skills

found_skills = extract_skills(text, skills)
# print(found_skills)

job_description = """
We are looking for a Machine Learning Engineer.

Required skills:
Python, Machine Learning, Deep Learning, NLP,
TensorFlow, Pandas, NumPy, SQL, Git and Docker.
"""

job_skills = extract_skills(job_description, skills)

job_skill_need = []
for skill in job_skills:
    job_skill_need.append(skill)
# print(job_skill_need)

resume_skills = extract_skills(text, skills)

matched_skills = list(set(found_skills) & set(job_skill_need))
missing_skills = list(set(job_skill_need) - set(found_skills))

# print("\nMatched skills:")
# for skill in matched_skills:
#     print(skill)

# print("\nMissing skills:")
# for skill in missing_skills:
#     print(skill)

match_score = (len(matched_skills) / len(job_skill_need)) * 100

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [text, job_description]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:2]
)

match_score = similarity[0][0] * 100

from gensim.models import Word2Vec

import gensim.downloader as api

wc = api.load('word2vec-google-news-300')

resume_tokens = text.lower().split()
job_tokens = job_description.lower().split()

sentences = [
    resume_tokens,
    job_tokens
]   

# model = wc(
#     sentences=sentences,
#     vector_size=100,
#     window=5,
#     min_count=1,
#     workers=4
# )
# print("Vocabulary:")
# print(model.wv.index_to_key)

# python_vector = model.wv["python"]

# print("Python vector:")
# print(python_vector)

# print("Vector shape:")
# print(python_vector.shape)