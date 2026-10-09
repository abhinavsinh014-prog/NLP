import pandas as pd
import numpy as np
import pymupdf

def extract_text_from_pdf(pdf_path):
    text = ""

    pdf = pymupdf.open(pdf_path)

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


text = extract_text_from_pdf("New blank-Converted.pdf")

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



import re

SKILLS = {
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++"],
    "sql": ["sql", "mysql", "postgresql"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural network", "neural networks"],
    "nlp": ["nlp", "natural language processing"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "git": ["git", "github"],
    "docker": ["docker"],
    "aws": ["aws", "amazon web services"],
    "streamlit": ["streamlit"],
    "opencv": ["opencv"],
    "word2vec": ["word2vec"],
}


