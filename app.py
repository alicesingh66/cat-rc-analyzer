import re
import nltk
import textstat
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Helper Functions ---
def clean_text(text):
    """Remove messy spaces/newlines"""
    return re.sub(r'\s+', ' ', text).strip()

def analyze_text(text):
    text = clean_text(text)
    
    if not text:
        return None, "Please enter some text!"
    
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    content_words = [w for w in words if w not in stop_words]
    
    lexical_density = len(content_words) / len(words) if words else 0
    flesch = textstat.flesch_reading_ease(text)
    fk_grade = textstat.flesch_kincaid_grade(text)
    
    if fk_grade < 10:
        level = "Easy (Light RC)"
    elif 10 <= fk_grade < 12:
        level = "Moderate (Medium RC)"
    elif 12 <= fk_grade < 14:
        level = "Hard (Dense RC)"
    elif 14 <= fk_grade < 16:
        level = "Very Hard (Philosophy/Academic RC)"
    else:
        level = "Extreme (Journal/Research RC)"
    
    result = {
        "Total Words": len(words),
        "Lexical Density": f"{lexical_density:.3f}",
        "Flesch Reading Ease": f"{flesch:.2f}",
        "Flesch-Kincaid Grade": fk_grade,
        "Difficulty Level": level
    }
    
    return result, None

# --- Streamlit UI ---
st.set_page_config(page_title="CAT RC Analyzer", layout="centered")
st.title("📊 CAT RC Analyzer")

text_input = st.text_area("Paste your RC passage below:", height=200)

if st.button("Analyze RC"):
    results, error = analyze_text(text_input)
    
    if error:
        st.warning(error)
    else:
        st.markdown("### 📝 Analysis Result")
        for key, value in results.items():
            st.write(f"**{key}:** {value}")


