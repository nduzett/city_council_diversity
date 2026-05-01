# -*- coding: utf-8 -*-
"""
01_preprocess.py
----------------
Loads raw LocalView parquet files, applies text preprocessing
(pause removal, lowercasing, punctuation removal, proper-noun removal,
lemmatization), and saves a cleaned CSV for downstream pipeline steps.

Output: SAVE_DIRECTORY / localview_data_preprocessed.csv
"""

import time
import re

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy
from tqdm import tqdm

from config import (
    DATA_DIRECTORY, SAVE_DIRECTORY,
    RAW_MEETINGS_TEMPLATE, PREPROCESSED_FILE,
    PREPROCESS_YEARS, MIN_DOCUMENT_LENGTH,
)

# ---------------------------------------------------------------------------
# One-time downloads / model loads
# ---------------------------------------------------------------------------
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

wl  = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
tqdm.pandas()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_wordnet_pos(tag: str):
    """Map a Penn-treebank POS tag to a WordNet POS constant."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def remove_proper_nouns(text: str) -> str:
    """Strip tokens tagged as proper nouns using spaCy (handles large docs)."""
    length = len(text)
    if length < 1_000_000:
        doc = nlp(text)
        return ' '.join(token.text for token in doc if token.pos_ != 'PROPN')
    elif length < 2_000_000:
        doc1 = nlp(text[:1_000_000])
        doc2 = nlp(text[1_000_001:])
        t1 = ' '.join(token.text for token in doc1 if token.pos_ != 'PROPN')
        t2 = ' '.join(token.text for token in doc2 if token.pos_ != 'PROPN')
        return t1 + " " + t2
    return ' '


def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)


def remove_pauses(text: str) -> str:
    """Remove bracketed pause annotations, e.g. [laughter]."""
    return re.sub(r"[\(\[].*?[\)\]]", "", text)


def lemmatizer(string_var: str) -> str:
    word_pos_tags = nltk.pos_tag(word_tokenize(string_var))
    lemmas = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in word_pos_tags]
    return " ".join(lemmas)


# ---------------------------------------------------------------------------
# Main pipeline step
# ---------------------------------------------------------------------------

def run_preprocessing():
    t0 = time.time()

    # -- Load and concatenate yearly parquet files ---------------------------
    frames = []
    for year in PREPROCESS_YEARS:
        path = f"{DATA_DIRECTORY}\\{RAW_MEETINGS_TEMPLATE.format(year=year)}"
        print(f"Appending {year}...")
        df_year = pd.read_parquet(path, engine='pyarrow')
        df_year = df_year[df_year['place_govt'] == 'MUNICIPAL COUNCIL']
        frames.append(df_year)

    meetings = pd.concat(frames, ignore_index=True)
    meetings['document'] = meetings['caption_text_clean']

    # -- Text preprocessing --------------------------------------------------
    print("\nPreprocessing...")

    print("  removing pauses")
    meetings['document'] = meetings['document'].progress_apply(remove_pauses)

    print("  converting to lowercase")
    meetings['document'] = meetings['document'].progress_apply(str.lower)

    print("  removing punctuation")
    meetings['document'] = meetings['document'].progress_apply(remove_punctuation)

    print("  removing blank captioning")
    meetings['document'] = meetings['document'].progress_apply(
        lambda x: x.replace("<no caption available>", ""))

    print("  removing proper nouns")
    meetings['document'] = meetings['document'].progress_apply(remove_proper_nouns)

    print("  lemmatizing")
    meetings['document'] = meetings['document'].progress_apply(lemmatizer)

    # -- Filter and save -----------------------------------------------------
    print("\nRemoving blank meetings and saving...")
    meetings = meetings[meetings['document'].str.len() > MIN_DOCUMENT_LENGTH]
    meetings['meeting_length'] = meetings['document'].apply(
        lambda x: len(str(x).split()))
    meetings['meeting_id'] = meetings.index

    print(f"Reduced to {len(meetings)} meetings.")

    save_df = meetings[['meeting_id', 'state_name', 'place_name',
                         'meeting_date', 'document', 'meeting_length']]
    out_path = f"{SAVE_DIRECTORY}\\{PREPROCESSED_FILE}"
    save_df.to_csv(out_path, index=False, header=True)
    print(f"Saved: {out_path}")

    elapsed = round((time.time() - t0) / 3600, 2)
    print(f"Preprocessing took {elapsed} hours")


if __name__ == "__main__":
    run_preprocessing()
