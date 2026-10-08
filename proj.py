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


def extract_skills(text, skills=SKILLS):
    """Return the set of canonical skill names found in `text`."""
    text = text.lower()
    found = set()

    for skill, aliases in skills.items():
        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, text):
                found.add(skill)
                break
    return found

job_description = """
We are looking for a Machine Learning Engineer.
Required skills: Python, Machine Learning, Deep Learning, NLP,
TensorFlow, Pandas, NumPy, SQL, Git and Docker.
"""

def compare_skills(resume_text, job_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    extra = resume_skills - job_skills

    score = len(matched) / len(job_skills) * 100 if job_skills else 0.0
    result = f"Matched skills: {sorted(matched)}\nMissing skills: {sorted(missing)}\nExtra skills: {sorted(extra)}\nMatch score: {round(score, 1)}%"
    return result


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_score(resume_text, job_text):
    """Return similarity (0-100) between resume and job description."""
    resume_clean = " ".join(preprocess(resume_text))
    job_clean = " ".join(preprocess(job_text))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([resume_clean, job_clean])

    similarity = cosine_similarity(matrix[0], matrix[1])[0][0]
    return round(similarity * 100, 1)


def job_keywords(job_text, resume_text, top_n=10):
    """Highest-weighted job terms that do NOT appear in the resume."""
    resume_clean = " ".join(preprocess(resume_text))
    job_clean = " ".join(preprocess(job_text))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([resume_clean, job_clean])

    terms = vectorizer.get_feature_names_out()
    job_weights = matrix[1].toarray()[0]
    resume_weights = matrix[0].toarray()[0]

    gaps = [
        (terms[i], job_weights[i])
        for i in range(len(terms))
        if job_weights[i] > 0 and resume_weights[i] == 0
    ]
    gaps.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _ in gaps[:top_n]]

import re
from sentence_transformers import SentenceTransformer, util

_model = None

def get_model():
    """Load the model once and reuse it (loading is slow)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def split_into_chunks(text, min_words=3):
    """Split text into sentence-sized pieces (lines, sentences, bullets)."""
    pieces = re.split(r"[\n.•●▪]+", text)
    return [p.strip() for p in pieces if len(p.split()) >= min_words]


def semantic_score(resume_text, job_text):
    """
    For each job requirement, find the most similar part of the resume.
    Return the average of those best matches (0-100) plus the details.
    """
    model = get_model()
    resume_chunks = split_into_chunks(resume_text)
    job_chunks = split_into_chunks(job_text)

    if not resume_chunks or not job_chunks:
        return 0.0, []

    resume_emb = model.encode(resume_chunks, convert_to_tensor=True)
    job_emb = model.encode(job_chunks, convert_to_tensor=True)

    sims = util.cos_sim(job_emb, resume_emb)      # shape: (job, resume)
    best = sims.max(dim=1)                        # best resume match per job chunk

    
