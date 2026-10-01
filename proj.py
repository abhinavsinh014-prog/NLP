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
print(found_skills)

