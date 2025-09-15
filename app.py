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



from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from collections import Counter
from textblob import TextBlob

# --- Hard Words & Meaning ---
def get_hard_words(text, top_n=10):
    words = tokenizer.tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stopwords.words('english')]
    hard_words = [w for w in words if len(w) >= 7]  # simple length filter
    freq = Counter(hard_words)
    top_words = freq.most_common(top_n)
    word_meanings = {}
    for w, _ in top_words:
        syns = wn.synsets(w)
        meaning = syns[0].definition() if syns else "Meaning not found"
        word_meanings[w] = meaning
    return word_meanings

# --- Tone ---
def get_tone(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# --- Central Idea ---
def get_central_idea(text):
    sentences = sent_tokenize(text)
    words = [w.lower() for w in tokenizer.tokenize(text) if w.isalpha() and w not in stopwords.words('english')]
    freq = Counter(words)
    # score sentences by sum of content word frequency
    scored = [(s, sum(freq.get(w.lower(),0) for w in tokenizer.tokenize(s) if w.isalpha())) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else "Could not determine"

# --- Structure (simple guess) ---
def get_structure(text):
    sentences = sent_tokenize(text)
    if any(w in text.lower() for w in ["because", "therefore", "however", "thus"]):
        return "Argumentative/Expository"
    elif len(sentences) > 8:
        return "Descriptive/Expository"
    else:
        return "Narrative/Short"
