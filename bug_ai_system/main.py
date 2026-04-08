# ==============================
# Bug AI System — Data Cleaning + Vectorization
# Author: You
# Purpose: Load bug reports, clean text, remove noise, and convert to numeric features for clustering
# ==============================

# Import required libraries
import pandas as pd           # Data manipulation
import string                 # For punctuation removal
import nltk                   # Natural Language Processing library
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numeric vectors

# ------------------------------
# Step 0: Download NLTK resources (only first time)
# ------------------------------
nltk.download('stopwords')    # Stopwords list for English
nltk.download('wordnet')      # WordNet for lemmatization

# ------------------------------
# Step 1: Load bug report data
# ------------------------------
df = pd.read_csv("bugs.csv")  # Load CSV containing bug reports

# ------------------------------
# Step 2: Stage 1 Cleaning — Filtering
# ------------------------------
df = df.drop_duplicates()           # Remove duplicate bug entries
df = df[df['description'].notna()]  # Remove rows with missing descriptions

# ------------------------------
# Step 3: Stage 2 Cleaning — Text Normalization
# ------------------------------
def normalize_text(text):
    """
    Convert text to lowercase, remove punctuation, and extra spaces.
    This prepares text for NLP processing.
    """
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Apply normalization to bug descriptions
df['clean_desc'] = df['description'].apply(normalize_text)

# ------------------------------
# Step 4: Stage 3 Cleaning — NLP (Stopwords + Lemmatization)
# ------------------------------
stop_words = set(stopwords.words('english'))   # English stopwords
lemmatizer = WordNetLemmatizer()               # Lemmatizer object

def nlp_clean(text):
    """
    Remove stopwords and lemmatize each word.
    Converts words to base forms and reduces noise.
    """
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Apply NLP cleaning
df['clean_desc'] = df['clean_desc'].apply(nlp_clean)

# Print cleaned data
print("After Stage 2 + 3 Cleaning:")
print(df[['description', 'clean_desc']])

# ------------------------------
# Step 5: Stage 4 — Feature Vectorization
# ------------------------------
vectorizer = TfidfVectorizer()   # Convert text into numeric TF-IDF vectors
X = vectorizer.fit_transform(df['clean_desc'])  # Matrix of shape (num_bugs, num_unique_words)

# Print feature matrix shape for verification
print("\nFeature Vector Shape:", X.shape)
