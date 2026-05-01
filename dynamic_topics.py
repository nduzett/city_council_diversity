# -*- coding: utf-8 -*-
"""
06_dynamic_topics.py
--------------------
Fits a BERTopic model on preprocessed meetings, then computes per-city
topic novelty scores (cosine distance from 2-year rolling mean) and
counts of newly-appearing topics within each city-state.

Input:  SAVE_DIRECTORY / localview_data_preprocessed.csv
Output: SAVE_DIRECTORY / topic_keywords_top10.txt
        SAVE_DIRECTORY / meeting_topic_novelty_by_city.csv
"""

import os

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

from config import (
    SAVE_DIRECTORY,
    PREPROCESSED_FILE, DYNAMIC_TOPIC_KEYWORDS, DYNAMIC_TOPIC_NOVELTY,
    DYN_MIN_DOC_FREQ, DYN_MAX_DOC_FREQ, DYN_MIN_MEETING_LENGTH,
    DYN_HDBSCAN_MIN_CLUSTER, DYN_NOVELTY_WINDOW_YEARS,
    DYN_TOPIC_THRESHOLD, DYN_MORE_STOP_WORDS,
)


# ---------------------------------------------------------------------------
# Main pipeline step
# ---------------------------------------------------------------------------

def run_dynamic_topics() -> None:
    # -- Load and filter data ------------------------------------------------
    print("Loading data...")
    df = pd.read_csv(f"{SAVE_DIRECTORY}\\{PREPROCESSED_FILE}", header=0)

    print("Cleaning data...")
    df['meeting_length'] = df['document'].apply(lambda x: len(str(x).split()))
    df = df[df['meeting_length'] >= DYN_MIN_MEETING_LENGTH].copy()
    df['meeting_date'] = df['meeting_date'].astype(str)
    df['meeting_year'] = df['meeting_date'].apply(lambda x: int(x.split("-")[0]))

    stop_words = list(stopwords.words('english')) + DYN_MORE_STOP_WORDS
    df = df[["document", "meeting_date", "meeting_year",
             "place_name", "state_name"]].copy()
    df["city_state"] = df["place_name"] + ", " + df["state_name"]

    # -- Vectorize and fit BERTopic ------------------------------------------
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        min_df=DYN_MIN_DOC_FREQ,
        max_df=DYN_MAX_DOC_FREQ
    )
    embeddings = vectorizer.fit_transform(df['document'])

    print("Fitting BERTopic model...")
    umap_model   = UMAP(random_state=42, low_memory=True)
    hdbscan_model = HDBSCAN(
        min_cluster_size=DYN_HDBSCAN_MIN_CLUSTER,
        prediction_data=True
    )
    topic_model = BERTopic(
        calculate_probabilities=True,
        hdbscan_model=hdbscan_model,
        umap_model=umap_model,
        low_memory=True
    )
    topics, probs = topic_model.fit_transform(df['document'], embeddings)

    # -- Save topic keywords -------------------------------------------------
    keywords_path = os.path.join(SAVE_DIRECTORY, DYNAMIC_TOPIC_KEYWORDS)
    with open(keywords_path, "w", encoding="utf-8") as f:
        for topic_id in topic_model.get_topics():
            words_scores = topic_model.get_topic(topic_id)
            if words_scores is None:
                continue
            f.write(f"Topic {topic_id}:\n")
            for word, score in words_scores[:10]:
                f.write(f"  {word}: {score:.4f}\n")
            f.write("\n")
    print(f"Topic keywords saved to: {keywords_path}")

    # -- Build doc_info with topic probability columns -----------------------
    topic_probs = np.stack(np.array(probs, dtype=object))
    num_topics  = topic_probs.shape[1]

    doc_info = pd.DataFrame({
        "meeting_date":  df["meeting_date"].values,
        "meeting_year":  df["meeting_year"].values,
        "place_name":    df["place_name"].values,
        "state_name":    df["state_name"].values,
        "city_state":    df["city_state"].values,
    })

    topic_share_cols = [f"topic_{i}_share" for i in range(num_topics)]
    topic_share_df   = pd.DataFrame(topic_probs, columns=topic_share_cols)
    doc_info         = pd.concat([doc_info, topic_share_df], axis=1)

    # -- Novelty scores (cosine distance to 2-year rolling mean) ------------
    print("Calculating novelty scores within each city-state...")
    novelty_scores = []
    similarities   = []

    for i in range(len(doc_info)):
        if i % 100 == 0:
            print(f"  {i}/{len(doc_info)}")

        current_year       = doc_info.loc[i, "meeting_year"]
        current_city_state = doc_info.loc[i, "city_state"]
        current_vec        = topic_probs[i].reshape(1, -1)

        window_start = current_year - DYN_NOVELTY_WINDOW_YEARS
        in_window = (
            (doc_info["city_state"] == current_city_state) &
            (doc_info["meeting_year"] >= window_start) &
            (doc_info["meeting_year"] < current_year)
        )
        earlier_vecs = topic_probs[in_window]

        if earlier_vecs.shape[0] == 0:
            similarities.append(None)
            novelty_scores.append(None)
            continue

        mean_vec = earlier_vecs.mean(axis=0).reshape(1, -1)
        sim      = cosine_similarity(current_vec, mean_vec)[0][0]
        similarities.append(sim)
        novelty_scores.append(1 - sim)

    doc_info["novelty_score_2yr_city"] = novelty_scores

    # -- Count newly-appearing topics per meeting ----------------------------
    print("Counting new topics within each city-state...")
    doc_info["topic"] = topics

    # Step 1: collect topic sets by (city_state, year)
    city_year_topicsets: dict = {}
    for (cs, yr), group in doc_info.groupby(["city_state", "meeting_year"]):
        all_topics: set = set()
        for i in group.index:
            active = set(np.where(topic_probs[i] > DYN_TOPIC_THRESHOLD)[0]) - {-1}
            all_topics.update(active)
        city_year_topicsets[(cs, yr)] = all_topics

    # Step 2: identify genuinely new topics (not seen in prior window years)
    new_topics_by_city_year: dict = {}
    for (cs, yr) in city_year_topicsets:
        prior: set = set()
        for y in range(yr - DYN_NOVELTY_WINDOW_YEARS, yr):
            prior.update(city_year_topicsets.get((cs, y), set()))
        new_topics_by_city_year[(cs, yr)] = city_year_topicsets[(cs, yr)] - prior

    # Step 3: count new topics per meeting
    new_topic_counts = []
    for i, row in doc_info.iterrows():
        if i % 100 == 0:
            print(f"  Processing meeting {i+1:,} of {len(doc_info):,}")
        if pd.isna(row["novelty_score_2yr_city"]):
            new_topic_counts.append(None)
            continue
        active_topics = set(np.where(topic_probs[i] > DYN_TOPIC_THRESHOLD)[0]) - {-1}
        new_topics    = new_topics_by_city_year.get(
            (row["city_state"], row["meeting_year"]), set())
        new_topic_counts.append(len(active_topics & new_topics))

    doc_info["num_new_topics_city"] = new_topic_counts

    # -- Save ----------------------------------------------------------------
    output_path = os.path.join(SAVE_DIRECTORY, DYNAMIC_TOPIC_NOVELTY)
    out_cols    = (["meeting_date", "meeting_year", "place_name",
                    "state_name", "city_state",
                    "novelty_score_2yr_city", "num_new_topics_city"]
                   + topic_share_cols)
    doc_info.to_csv(output_path, index=False, columns=out_cols)
    print(f"Dynamic topic novelty saved to: {output_path}")


if __name__ == "__main__":
    run_dynamic_topics()
