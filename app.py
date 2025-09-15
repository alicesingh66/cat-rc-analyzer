# import re
# import nltk
# import textstat
# import streamlit as st
# from nltk.corpus import stopwords
# from nltk.tokenize import TreebankWordTokenizer

# # --- Setup NLTK stopwords ---
# nltk.download('stopwords', quiet=True)

# # Initialize tokenizer
# tokenizer = TreebankWordTokenizer()

# # --- Helper Functions ---
# def clean_text(text):
#     """Remove messy spaces/newlines"""
#     return re.sub(r'\s+', ' ', text).strip()

# def analyze_text(text):
#     text = clean_text(text)
    
#     if not text:
#         return None, "Please enter some text!"
    
#     # Use Treebank tokenizer instead of word_tokenize
#     words = tokenizer.tokenize(text.lower())
#     words = [w for w in words if w.isalpha()]
    
#     stop_words = set(stopwords.words('english'))
#     content_words = [w for w in words if w not in stop_words]
    
#     lexical_density = len(content_words) / len(words) if words else 0
#     flesch = textstat.flesch_reading_ease(text)
#     fk_grade = textstat.flesch_kincaid_grade(text)
    
#     if fk_grade < 10:
#         level = "Easy (Light RC)"
#     elif 10 <= fk_grade < 12:
#         level = "Moderate (Medium RC)"
#     elif 12 <= fk_grade < 14:
#         level = "Hard (Dense RC)"
#     elif 14 <= fk_grade < 16:
#         level = "Very Hard (Philosophy/Academic RC)"
#     else:
#         level = "Extreme (Journal/Research RC)"
    
#     result = {
#         "Total Words": len(words),
#         "Lexical Density": f"{lexical_density:.3f}",
#         "Flesch Reading Ease": f"{flesch:.2f}",
#         "Flesch-Kincaid Grade": fk_grade,
#         "Difficulty Level": level
#     }
    
#     return result, None

# # --- Streamlit UI ---
# st.set_page_config(page_title="CAT RC Analyzer", layout="centered")
# st.title("📊 CAT RC Analyzer")

# text_input = st.text_area("Paste your RC passage below:", height=200)

# if st.button("Analyze RC"):
#     results, error = analyze_text(text_input)
    
#     if error:
#         st.warning(error)
#     else:
#         st.markdown("### 📝 Analysis Result")
#         for key, value in results.items():
#             st.write(f"**{key}:** {value}")

import re
import streamlit as st
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
from textblob import TextBlob
import textstat
import pandas as pd
import openai

# --- NLTK downloads ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- Initialize tokenizer ---
tokenizer = TreebankWordTokenizer()

# --- OpenAI API key via Streamlit secrets ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Helper Functions ---
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def simple_sent_tokenize(text):
    """Split text into sentences without NLTK Punkt"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def analyze_metrics(text):
    text = clean_text(text)
    words = tokenizer.tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    content_words = [w for w in words if w not in stop_words]

    lexical_density = len(content_words) / len(words) if words else 0
    flesch = textstat.flesch_reading_ease(text)
    fk_grade = textstat.flesch_kincaid_grade(text)

    if fk_grade < 10:
        level = "Easy (Light RC)"
        color = "#4CAF50"
    elif 10 <= fk_grade < 12:
        level = "Moderate (Medium RC)"
        color = "#FFC107"
    elif 12 <= fk_grade < 14:
        level = "Hard (Dense RC)"
        color = "#FF5722"
    elif 14 <= fk_grade < 16:
        level = "Very Hard (Academic RC)"
        color = "#E91E63"
    else:
        level = "Extreme (Research RC)"
        color = "#9C27B0"

    return {
        "Total Words": len(words),
        "Lexical Density": f"{lexical_density:.3f}",
        "Flesch Reading Ease": f"{flesch:.2f}",
        "Flesch-Kincaid Grade": fk_grade,
        "Difficulty Level": (level, color)
    }

def get_hard_words(text, top_n=10):
    words = tokenizer.tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stopwords.words('english')]
    hard_words = [w for w in words if len(w) >= 7]
    freq = Counter(hard_words)
    top_words = freq.most_common(top_n)
    word_meanings = {}
    for w, _ in top_words:
        syns = wn.synsets(w)
        meaning = syns[0].definition() if syns else "Meaning not found"
        word_meanings[w] = meaning
    return word_meanings

def get_tone(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def get_central_idea(text):
    sentences = simple_sent_tokenize(text)
    words = [w.lower() for w in tokenizer.tokenize(text) if w.isalpha() and w not in stopwords.words('english')]
    freq = Counter(words)
    scored = [(s, sum(freq.get(w.lower(),0) for w in tokenizer.tokenize(s) if w.isalpha())) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else "Could not determine"

def get_structure(text):
    sentences = simple_sent_tokenize(text)
    if any(w in text.lower() for w in ["because", "therefore", "however", "thus"]):
        return "Argumentative/Expository"
    elif len(sentences) > 8:
        return "Descriptive/Expository"
    else:
        return "Narrative/Short"

# --- AI Analysis ---
def ai_analyze_rc(passage):
    prompt = f"""
    Analyze the following CAT Reading Comprehension passage.

    1. Central idea in one sentence
    2. Tone (critical, neutral, formal, etc.)
    3. Structure (narrative, expository, argumentative)
    4. Extract 5 difficult words with meanings in context
    5. Suggest 3 CAT-style questions with answers

    Passage:
    {passage}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]

# --- Streamlit UI ---
st.set_page_config(page_title="CAT RC Analyzer", layout="wide")
st.title("📊 CAT RC Analyzer - Full RC Insights")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Paste your RC passage in the text box.
2. Click 'Analyze RC'.
3. View metrics, hard words, tone, central idea, structure, and AI insights.
""")

text_input = st.text_area("Paste your RC passage below:", height=200)

if st.button("Analyze RC"):
    if not text_input.strip():
        st.warning("Please enter some text!")
    else:
        # --- Rule-based Metrics ---
        metrics = analyze_metrics(text_input)
        col1, col2, col3 = st.columns(3)
        col1.metric("📝 Total Words", metrics["Total Words"])
        col2.metric("📖 Lexical Density", metrics["Lexical Density"])
        col3.metric("📗 Flesch Reading Ease", metrics["Flesch Reading Ease"])

        col4, col5 = st.columns(2)
        col4.metric("🎓 FK Grade", metrics["Flesch-Kincaid Grade"])
        level, color = metrics["Difficulty Level"]
        col5.markdown(
            f"<div style='padding:10px; background-color:{color}; color:white; text-align:center; border-radius:5px;'>🔥 {level}</div>",
            unsafe_allow_html=True
        )

        # --- Top Hard Words ---
        hard_words = get_hard_words(text_input)
        if hard_words:
            st.subheader("Top 10 Hard Words & Meanings")
            st.table(pd.DataFrame(list(hard_words.items()), columns=["Word", "Meaning"]))

        # --- Tone ---
        tone = get_tone(text_input)
        st.subheader("Tone of Passage (Rule-based)")
        st.write(f"**{tone}**")

        # --- Central Idea ---
        central_idea = get_central_idea(text_input)
        st.subheader("Central Idea (Rule-based)")
        st.write(central_idea)

        # --- Structure ---
        structure = get_structure(text_input)
        st.subheader("Passage Structure (Rule-based Approx.)")
        st.write(structure)

        # --- AI Analysis ---
        with st.spinner("Generating AI-based analysis..."):
            ai_result = ai_analyze_rc(text_input)
            st.subheader("AI-Based Full Analysis")
            st.write(ai_result)




