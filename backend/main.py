from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")
app = FastAPI()

class ResumeText(BaseModel):
    text: str

@app.post("/analyze/")
def analyze_resume(resume: ResumeText):
    text = resume.text

    # --- Sentiment Analysis ---
    blob = TextBlob(text)
    sentiment = {
        "VADER": {"compound": 0.5, "pos": 0.6, "neg": 0.2, "neu": 0.2},  # replace with real VADER if used
        "Polarity": blob.sentiment.polarity,
        "Subjectivity": blob.sentiment.subjectivity
    }

    # --- Passive Voice Detection ---
    doc = nlp(text)
    passive_sents = [sent.text for sent in doc.sents if any(token.dep_ == "auxpass" for token in sent)]

    # --- Simple assertive suggestions (example) ---
    suggestions = ["Replace weak phrases with action verbs"] if "responsible for" in text else []

    return {
        "sentiment": sentiment,
        "passive_sents": passive_sents,
        "suggestions": suggestions
    }
