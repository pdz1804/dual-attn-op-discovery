import re
import spacy
import pandas as pd
from nltk.corpus import stopwords as sw
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

tqdm.pandas()          # Enable for pandas

# === Load spaCy for lemmatization ===
# try:
#     logger.info("Loading spaCy model 'en_core_web_sm'...")
#     # nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
#     logger.info("spaCy model loaded successfully.")
# except Exception as e:
#     logger.error(f"Failed to load spaCy model: {e}")
#     raise e

# Build stopword set (NLTK + common boilerplate/URL remnants)
stop_words = set(sw.words("english"))
stop_words.update({"ltd", "inc", "www", "com", "http", "https", "org", "net", "edu", "gov", "co", "uk", "us", "ca", "jp", "fr", "de", "html", "php", "asp"})

# Compile your regexes once
URL_DOTTED = re.compile(r'^(?:https?://)?(?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/.*)?$', re.IGNORECASE)
URL_GLUED  = re.compile(r'^www[a-z0-9\-]+$', re.IGNORECASE)
NON_ASCII  = re.compile(r'[^\x00-\x7F]')
HAS_DIGIT  = re.compile(r'\d')

# Boilerplate stop-tokens
BOILERPLATE = {
    'ltd','inc','www','com','http','https',
}

# Days of the week and months
TIME_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august",
    "sep", "sept", "september", "oct", "october",
    "nov", "november", "dec", "december"
}

def clean_tokens(text):
    original_tokens = text.split('|')
    out = []
    filtered_count = 0

    for tok in original_tokens:
        t = tok.strip().lower()
        if not t:
            continue

        # RL/domain?
        if URL_DOTTED.match(t) or URL_GLUED.match(t):
            continue

        # non-ASCII?
        if NON_ASCII.search(t):
            continue

        # any digit?
        if HAS_DIGIT.search(t):
            continue

        # Remove short, boilerplate, time-related tokens
        if t in BOILERPLATE or t in TIME_WORDS or len(t) < 2:
            continue

        if t.endswith("com"):
            continue

        out.append(t)

    logger.debug(f"Processed text: {len(original_tokens)} tokens → {len(out)} kept, {filtered_count} filtered.")

    return "|".join(out)

    # Lemmatize filtered tokens
    # doc = nlp(" ".join(out))  # Create a spaCy doc for lemmatization
    # lemmas = [token.lemma_ for token in doc if token.lemma_ not in stop_words and len(token.lemma_) > 1]
    


