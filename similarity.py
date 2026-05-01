# -*- coding: utf-8 -*-
"""
05_similarity.py
----------------
Computes pairwise TF-IDF cosine similarity between each meeting and all
prior meetings in the same city/state, then cleans the output into a
long-format CSV.

Steps
-----
1. embed  – build TF-IDF vector per meeting  → meeting_embeddings.csv
2. similarity – compute cosine sim row-by-row → meeting_similarity_data.csv
3. clean  – explode pipe-delimited scores     → meeting_similarity_data_split.csv

Input:  SAVE_DIRECTORY / localview_data_preprocessed.csv
        (and election_places.csv in DATA_DIRECTORY for the merge step)
Output: see above
"""

import gc

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from config import (
    DATA_DIRECTORY, SAVE_DIRECTORY,
    PREPROCESSED_FILE, ELECTION_PLACES_FILE,
    EMBEDDINGS_FILE, SIMILARITY_RAW_FILE, SIMILARITY_SPLIT_FILE,
    SIMILARITY_STARTING_ROW,
    SIMILARITY_TFIDF_MAX_DF, SIMILARITY_TFIDF_MIN_DF,
)

tqdm.pandas()

# ---------------------------------------------------------------------------
# Helper: election-place merge (keeps only meetings that match election data)
# ---------------------------------------------------------------------------

_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "District of Columbia": "DC", "Delaware": "DE", "Florida": "FL",
    "Georgia": "GA", "Hawaii": "HI", "Iowa": "IA", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Massachusetts": "MA", "Maryland": "MD", "Maine": "ME",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "North Carolina": "NC", "North Dakota": "ND",
    "Nebraska": "NE", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "Nevada": "NV", "New York": "NY", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Virginia": "VA",
    "Vermont": "VT", "Washington": "WA", "Wisconsin": "WI",
    "West Virginia": "WV", "Wyoming": "WY",
}

_PLACE_SUFFIXES = [" city", " town", " village"]


def _clean_place(s: str) -> str:
    s = s.lower().strip()
    for suffix in _PLACE_SUFFIXES:
        s = s.replace(suffix, "")
    return s


def merge_election_places(df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join *df* to the election-places lookup table."""
    election_places = pd.read_csv(
        f"{DATA_DIRECTORY}\\{ELECTION_PLACES_FILE}")

    # Normalise state
    for full, abbr in _STATE_ABBR.items():
        df['state_name'] = df['state_name'].str.replace(full, abbr,
                                                         regex=False)

    df['place_name']              = df['place_name'].apply(_clean_place)
    election_places['geo_name']   = election_places['geo_name'].apply(_clean_place)

    merged = pd.merge(
        election_places, df,
        left_on=['state_abb', 'geo_name'],
        right_on=['state_name', 'place_name'],
        how='inner'
    )
    return merged


# ---------------------------------------------------------------------------
# Helper: unique word count
# ---------------------------------------------------------------------------

def count_unique_words(text: str) -> int:
    return len(set(text.split()))


# ---------------------------------------------------------------------------
# Step 1 – TF-IDF embedding
# ---------------------------------------------------------------------------

def vocab_embed(documents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a TF-IDF vectorizer, then store each document's vector as a
    pipe-delimited string in a new 'tfidf_vector' column.
    """
    documents = documents_df['document']
    print("Fitting TF-IDF vectorizer...")
    vectorizer   = TfidfVectorizer(
        max_df=SIMILARITY_TFIDF_MAX_DF,
        min_df=SIMILARITY_TFIDF_MIN_DF,
        stop_words="english",
        binary=False
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_vectors = ["|".join(map(str, row.toarray().flatten()))
                     for row in tfidf_matrix]
    result = documents_df.copy()
    result["tfidf_vector"] = tfidf_vectors
    print("TF-IDF vectors added.")
    return result


def run_embedding() -> None:
    """Merge election places, embed, and save."""
    df = pd.read_csv(f"{SAVE_DIRECTORY}\\{PREPROCESSED_FILE}", header=0)
    merged_df = merge_election_places(df)
    del df

    print("Calculating meeting length and unique word counts...")
    merged_df['meeting_length']   = merged_df['document'].progress_apply(
        lambda x: len(str(x).split()))
    merged_df['unique_word_count'] = merged_df['document'].progress_apply(
        count_unique_words)

    print("Embedding...")
    save_df = vocab_embed(merged_df)

    out_path = f"{SAVE_DIRECTORY}\\{EMBEDDINGS_FILE}"
    save_df.to_csv(out_path, index=False, header=True)
    print(f"Embeddings saved to: {out_path}")

    del merged_df, save_df
    gc.collect()


# ---------------------------------------------------------------------------
# Step 2 – per-row similarity computation
# ---------------------------------------------------------------------------

def compute_similarity(index: int, df: pd.DataFrame,
                       starting_row: int, output_file: str) -> None:
    """Write one row of similarity scores to *output_file*."""
    current_date  = df.loc[index, "meeting_date_clean"]
    current_city  = df.loc[index, "place_name"]
    current_state = df.loc[index, "state_name"]

    past = df[
        (df["state_name"] == current_state) &
        (df["place_name"] == current_city) &
        (df["meeting_date_clean"] < current_date)
    ].copy()

    if past.empty:
        similarities, past_dates = [], []
    else:
        curr_vec = np.array(
            [float(x) for x in df.loc[index, "tfidf_vector"].split('|')]
        ).reshape(1, -1)

        past["np_array"] = past["tfidf_vector"].apply(
            lambda x: np.array([float(i) for i in x.split('|')]))
        past["similarity"] = past["np_array"].apply(
            lambda x: cosine_similarity(curr_vec, x.reshape(1, -1))[0][0])

        similarities = past["similarity"].apply(lambda x: f"{x:.4f}").tolist()
        past_dates   = past["meeting_date_clean"].astype(str).tolist()

    row_df = pd.DataFrame({
        "state_name":        [current_state],
        "place_name":        [current_city],
        "meeting_date_clean": [current_date],
        "meeting_length":    [df.loc[index, "meeting_length"]],
        "unique_word_count": [df.loc[index, "unique_word_count"]],
        "similarities":      ["|".join(similarities)],
        "past_meeting_dates": ["|".join(past_dates)],
    })

    write_header = (index + starting_row == 0)
    write_mode   = "w" if write_header else "a"
    row_df.to_csv(output_file, mode=write_mode, index=False, header=write_header)


def run_similarity(starting_row: int = SIMILARITY_STARTING_ROW) -> None:
    input_file  = f"{SAVE_DIRECTORY}\\{EMBEDDINGS_FILE}"
    output_file = f"{SAVE_DIRECTORY}\\{SIMILARITY_RAW_FILE}"

    full_df = pd.read_csv(
        input_file,
        usecols=["state_name", "place_name", "meeting_date",
                 "meeting_length", "unique_word_count", "tfidf_vector"],
        skiprows=range(1, starting_row + 1)
    )
    full_df["meeting_date_clean"] = pd.to_datetime(full_df["meeting_date"])
    full_df = full_df.sort_values(
        by=["state_name", "place_name", "meeting_date_clean"]
    ).reset_index(drop=True)

    full_df.progress_apply(
        lambda row: compute_similarity(row.name, full_df,
                                       starting_row, output_file), axis=1)
    print(f"Similarity data saved to: {output_file}")


# ---------------------------------------------------------------------------
# Step 3 – clean / explode similarity output
# ---------------------------------------------------------------------------

def clean_similarity() -> None:
    in_path  = f"{SAVE_DIRECTORY}\\{SIMILARITY_RAW_FILE}"
    out_path = f"{SAVE_DIRECTORY}\\{SIMILARITY_SPLIT_FILE}"

    df = pd.read_csv(in_path)

    print("Splitting...")
    df["sim_score"] = df["similarities"].str.split("|")
    df["sim_date"]  = df["past_meeting_dates"].str.split("|")

    print("Exploding...")
    df = df.explode(["sim_score", "sim_date"], ignore_index=True)

    save_df = df[["state_name", "place_name", "meeting_date_clean",
                  "meeting_length", "unique_word_count",
                  "sim_score", "sim_date"]]
    save_df.to_csv(out_path, mode="w", index=False, header=True)
    print(f"Split similarity data saved to: {out_path}")


if __name__ == "__main__":
    run_embedding()
    run_similarity(SIMILARITY_STARTING_ROW)
    clean_similarity()
