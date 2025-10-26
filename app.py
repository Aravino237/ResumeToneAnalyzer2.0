import streamlit as st
import PyPDF2
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import re

# -------------------------------
# Load SpaCy model
# -------------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# Initialize variables
# -------------------------------
text = ""
sentiment = {}
passive_sents = []
suggestions = []

# -------------------------------
# Helper Functions
# -------------------------------

# PDF report generator
def create_pdf_report(resume_text, sentiment, passive_sents, suggestions):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Resume Tone Analyzer Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Sentiment
    story.append(Paragraph("<b>Sentiment Analysis</b>", styles['Heading2']))
    vader = sentiment.get("VADER", {})
    story.append(Paragraph(f"VADER compound: {vader.get('compound', 0):.3f}", styles['Normal']))
    story.append(Paragraph(f"Pos/Neg/Neu: {vader.get('pos', 0):.2f}/{vader.get('neg', 0):.2f}/{vader.get('neu', 0):.2f}", styles['Normal']))
    story.append(Paragraph(f"TextBlob Polarity: {sentiment.get('Polarity', 0):.3f}, Subjectivity: {sentiment.get('Subjectivity', 0):.3f}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Passive sentences
    story.append(Paragraph("<b>Passive Voice Sentences</b>", styles['Heading2']))
    if passive_sents:
        for s in passive_sents:
            story.append(Paragraph(f"• {s}", styles['Normal']))
    else:
        story.append(Paragraph("No passive voice sentences detected.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Assertive suggestions
    story.append(Paragraph("<b>Assertive Rewording Suggestions</b>", styles['Heading2']))
    if suggestions:
        for s in suggestions:
            story.append(Paragraph(f"• {s}", styles['Normal']))
    else:
        story.append(Paragraph("No weak phrases detected.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Resume excerpt
    story.append(Paragraph("<b>Extracted Resume Text</b> (first 5000 chars)", styles['Heading2']))
    excerpt = resume_text[:5000].replace("\n", "<br/>")
    story.append(Paragraph(excerpt, styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# Text extraction from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Text extraction from TXT
def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")

# Passive voice detection
def detect_passive_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if any(token.dep_ == "auxpass" for token in sent)]

# Assertive suggestions
assertive_replacements = {
    "responsible for": "led",
    "assisted in": "collaborated on",
    "helped": "contributed to",
    "involved in": "participated in",
    "worked on": "developed",
    "participated in": "took initiative in"
}

def suggest_rewordings(text):
    suggestions = []
    lower_text = text.lower()
    for weak, strong in assertive_replacements.items():
        if weak in lower_text:
            suggestions.append(f"Replace '{weak}' → '{strong}'")
    return suggestions

# Highlight text for Streamlit display
def highlight_text(text):
    highlights = []
    for weak in assertive_replacements.keys():
        highlights.append({'pattern': weak, 'color': '#fff2b2'})  # Yellow
    for p in ["was", "were", "been", "being", "by"]:
        highlights.append({'pattern': p, 'color': '#ffd6d6'})  # Red

    html = text
    for item in highlights:
        pat = re.escape(item['pattern'])
        repl = f"<mark style='background:{item['color']};padding:0.1rem;border-radius:3px'>{item['pattern']}</mark>"
        html = re.sub(pat, repl, html, flags=re.IGNORECASE)
    html = html.replace("\n", "<br>")
    return f"<div style='font-family:sans-serif; line-height:1.6; font-size:15px'>{html}</div>"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Resume Tone Analyzer", layout="wide")

# CSS for colorful UI
# --- Custom CSS for black tabs ---
st.markdown("""
    <style>
    /* Style the tabs */
    .stTabs [data-baseweb="tab-list"] button {
        background-color: black !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        margin-right: 8px !important;
        transition: background-color 0.3s ease;
    }

    /* Highlight the active tab in gray */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #333333 !important;
        color: white !important;
    }

    /* Hover effect */
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #222222 !important;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Resume Tone Analyzer")
st.caption("Upload a resume and get tone, sentiment, and assertiveness analysis.")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Upload Resume", "Tone & Sentiment", "Passive Voice"])

# -------------------------------
# Tab 1: Upload
# -------------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_txt(uploaded_file)
        st.success("✅ Resume uploaded successfully!")

# -------------------------------
# Tab 2: Tone & Sentiment
# -------------------------------
with tab2:
    if text != "":
        analyzer = SentimentIntensityAnalyzer()
        vader_scores = analyzer.polarity_scores(text)
        blob = TextBlob(text)
        sentiment = {
            "VADER": vader_scores,
            "Polarity": blob.sentiment.polarity,
            "Subjectivity": blob.sentiment.subjectivity
        }

        col1, col2 = st.columns(2)
        with col1:
            st.metric("VADER Compound", f"{vader_scores['compound']:.3f}")
            st.metric("Pos / Neg / Neu", f"{vader_scores['pos']:.2f} / {vader_scores['neg']:.2f} / {vader_scores['neu']:.2f}")
        with col2:
            st.write(f"TextBlob Polarity: **{blob.sentiment.polarity:.3f}**")
            st.write(f"Subjectivity: **{blob.sentiment.subjectivity:.3f}**")

        passive_sents = detect_passive_sentences(text)
        suggestions = suggest_rewordings(text)

        # PDF download button
        pdf_bytes = create_pdf_report(text, sentiment, passive_sents, suggestions)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="resume_tone_report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Upload a resume in Tab 1 first to see analysis.")

# -------------------------------
# Tab 3: Passive Voice
# -------------------------------
with tab3:
    if text != "":
        passive_sents = detect_passive_sentences(text)
        suggestions = suggest_rewordings(text)
        st.subheader("🪶 Passive Voice Sentences")
        if passive_sents:
            st.warning("⚠️ Passive voice detected:")
            for s in passive_sents:
                st.markdown(f"• {s}")
        else:
            st.success("✅ No passive sentences detected.")
        st.subheader("✍️ Assertive Rewording Suggestions")
        if suggestions:
            for s in suggestions:
                st.markdown(f"• {s}")
        else:
            st.success("✅ No weak phrases detected.")
    else:
        st.info("Upload a resume in Tab 1 first.")