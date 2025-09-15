import re
import nltk
import textstat
import tkinter as tk
from tkinter import scrolledtext, messagebox
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (only first run)
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """Remove messy spaces/newlines"""
    return re.sub(r'\s+', ' ', text).strip()

def analyze_text():
    raw_text = text_box.get("1.0", tk.END)
    text = clean_text(raw_text)
    
    if not text:
        messagebox.showwarning("Warning", "Please enter some text!")
        return
    
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
    
    # Display results
    result = (
        "📊 CAT RC Passage Analysis\n"
        "===============================\n"
        f"📝 Total Words        : {len(words)}\n"
        f"📖 Lexical Density    : {lexical_density:.3f}\n"
        f"📗 Flesch Reading Ease: {flesch:.2f}\n"
        f"🎓 Flesch-Kincaid Gr. : {fk_grade}\n"
        f"🔥 Difficulty Level   : {level}\n"
        "===============================\n"
    )
    
    result_box.config(state="normal")
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, result)
    result_box.config(state="disabled")

# --- Tkinter UI setup ---
root = tk.Tk()
root.title("CAT RC Analyzer")
root.geometry("600x500")

tk.Label(root, text="Paste your RC passage below:", font=("Arial", 12)).pack(pady=5)

text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=12, font=("Arial", 11))
text_box.pack(padx=10, pady=5)

analyze_btn = tk.Button(root, text="Analyze RC", font=("Arial", 12, "bold"), command=analyze_text, bg="#4CAF50", fg="white")
analyze_btn.pack(pady=10)

result_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=10, font=("Courier", 11))
result_box.pack(padx=10, pady=5)
result_box.config(state="disabled")

root.mainloop()


