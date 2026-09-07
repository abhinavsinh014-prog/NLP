import nltk
from nltk.tokenize import sent_tokenize

resume = {
    "name": "John Doe",
    "role": "Software Engineer",
    "skills": ["Python", "JavaScript", "React", "Django"],
    "experience": ["2 years"]
}

text = f"""
Name: {resume['name']}
Role: {resume['role']}
Skills: {', '.join(resume['skills'])}
Experience: {resume['experience'][0]}
"""

doc = sent_tokenize(text)

print(doc)



