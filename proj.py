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
def extract_skills(text, skills):
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

def calculate_match_score(text, job_description):
    documents = [text, job_description]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    match_score = similarity[0][0] * 100

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()                              # 1. lowercase
    text = re.sub(r"http\S+|www\S+", " ", text)      # 2. remove links
    text = re.sub(r"\S+@\S+", " ", text)             # 3. remove emails
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)      # 4. remove symbols
    text = re.sub(r"\s+", " ", text).strip()         # 5. collapse spaces
    return text

def preprocess(text):
    text = clean_text(text)
    tokens = word_tokenize(text)                     # split into words
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

tokens = preprocess(text)
print(tokens)
