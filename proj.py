import pandas as pd
import numpy as np
import nltk as nltk
# nltk.download('punkt')
# nltk.download('stopwords')
import tensorflow as tf

import pymupdf


def extract_text_from_pdf(pdf_path):
    text = ""

    pdf = pymupdf.open(pdf_path)

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


text = extract_text_from_pdf("google cloud ecerti.pdf")

print(text)