# -*- coding: utf-8 -*-
"""
04_topics.py
------------
Fits an LDA topic model on preprocessed meeting documents, merges similar
topics, and saves per-document topic probability distributions.

Input:  SAVE_DIRECTORY / localview_data_preprocessed.csv
Output: SAVE_DIRECTORY / topic_data_final.csv
        SAVE_DIRECTORY / merged_topics.txt
"""

import time

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gc

from config import (
    SAVE_DIRECTORY,
    PREPROCESSED_FILE, TOPIC_DATA_FILE, TOPICS_MERGED_TXT,
    TOPIC_MIN_DOC_FREQ, TOPIC_MAX_DOC_FREQ,
    TOPIC_NUM_TOPICS, TOPIC_ALPHA, TOPIC_BETA, TOPIC_MAX_ITER,
    TOPIC_NUM_TOP_WORDS, TOPIC_MERGE_CUTOFF,
    TOPIC_MORE_STOP_WORDS, TOPIC_MIN_MEETING_LENGTH,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def merge_topics(lda_model, num_topics: int, threshold: float = 0.8) -> list:
    """
    Group topics whose cosine similarity exceeds *threshold*.

    Returns a list of groups, where each group is a list of original topic
    indices to be merged together.
    """
    topic_word_dist = (lda_model.components_ /
                       lda_model.components_.sum(axis=1)[:, np.newaxis])
    sim_matrix   = cosine_similarity(topic_word_dist)
    merged_groups = []
    visited       = set()

    for i in range(num_topics):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, num_topics):
            if j not in visited and sim_matrix[i, j] >= threshold:
                group.append(j)
                visited.add(j)
        merged_groups.append(group)

    return merged_groups


def merge_topic_distributions(merged_groups: list,
                               doc_topic_matrix: np.ndarray) -> np.ndarray:
    """Sum probabilities of original topics within each merged group."""
    new_matrix = np.zeros((doc_topic_matrix.shape[0], len(merged_groups)))
    for new_idx, group in enumerate(merged_groups):
        new_matrix[:, new_idx] = np.sum(doc_topic_matrix[:, group], axis=1)
    return new_matrix


def print_merged_topics(lda_model, vectorizer,
                        merged_groups: list, top_n_words: int) -> None:
    """Write merged topic keywords to a text file and stdout."""
    out_path = f"{SAVE_DIRECTORY}\\{TOPICS_MERGED_TXT}"
    words     = vectorizer.get_feature_names_out()
    topic_word_dist = (lda_model.components_ /
                       lda_model.components_.sum(axis=1)[:, np.newaxis])

    with open(out_path, "w") as f:
        print("Merged Topics:")
        f.write("Merged Topics:\n")
        for new_idx, group in enumerate(merged_groups):
            merged_dist    = np.sum(topic_word_dist[group], axis=0)
            top_word_idx   = merged_dist.argsort()[-top_n_words:][::-1]
            top_words      = [words[i] for i in top_word_idx]
            line           = f"Merged Topic {new_idx}: {', '.join(top_words)}"
            print(line)
            f.write(line + "\n")

    print(f"Merged topics saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main pipeline step
# ---------------------------------------------------------------------------

def get_topics(df: pd.DataFrame,
               min_doc_freq: float, max_doc_freq: float,
               num_topics: int, alpha: float, beta: float,
               num_top_words: int, merge_cutoff: float,
               more_stop_words: list) -> None:
    """
    Fit LDA, merge similar topics, and save document-level probability CSV.
    """
    df = df.copy()
    df['meeting_date'] = df['meeting_date'].apply(str)
    df['meeting_year'] = df['meeting_date'].apply(lambda x: int(x.split("-")[0]))

    documents  = df['document']
    stop_words = list(stopwords.words('english')) + more_stop_words

    # -- Vectorize -----------------------------------------------------------
    print("\nVectorizing...")
    t0 = time.time()
    vectorizer    = TfidfVectorizer(stop_words=stop_words,
                                    min_df=min_doc_freq, max_df=max_doc_freq)
    doc_term_matrix = vectorizer.fit_transform(documents)
    print(f"Vectorizing took {round((time.time() - t0) / 3600, 2)} hours")

    # -- Fit LDA -------------------------------------------------------------
    print("\nFitting LDA model...")
    t0 = time.time()
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        doc_topic_prior=alpha,
        topic_word_prior=beta,
        max_iter=TOPIC_MAX_ITER,
        random_state=42
    )
    lda_model.fit(doc_term_matrix)
    print(f"Fitting LDA took {round((time.time() - t0) / 3600, 2)} hours")

    # -- Merge similar topics ------------------------------------------------
    print("\nMerging similar topics...")
    merged_groups      = merge_topics(lda_model, num_topics, threshold=merge_cutoff)
    doc_topic_matrix   = lda_model.transform(doc_term_matrix)
    merged_doc_topic   = merge_topic_distributions(merged_groups, doc_topic_matrix)

    print_merged_topics(lda_model, vectorizer, merged_groups, num_top_words)

    # -- Save results --------------------------------------------------------
    print("\nSaving merged topic probabilities...")
    topic_df = pd.DataFrame(
        merged_doc_topic,
        columns=[f'merged_topic_{i}' for i in range(len(merged_groups))]
    )
    df['merge_string'] = df['document'].str[:50]
    save_df = df[['state_name', 'place_name', 'meeting_date',
                  'meeting_length', 'merge_string']].copy()

    for x in range(len(merged_groups)):
        save_df[f'topic_{x}_probs'] = topic_df[f'merged_topic_{x}']

    out_path = f"{SAVE_DIRECTORY}\\{TOPIC_DATA_FILE}"
    save_df.to_csv(out_path, index=False, header=True)
    print(f"Topic data saved to: {out_path}")

    print("\nTopic means:")
    for x in range(len(merged_groups)):
        print(f"  Topic {x}: {topic_df[f'merged_topic_{x}'].mean():.4f}")

    # -- Free memory ---------------------------------------------------------
    del df, save_df, topic_df, doc_topic_matrix, merged_doc_topic
    gc.collect()


if __name__ == "__main__":
    full_df = pd.read_csv(
        f"{SAVE_DIRECTORY}\\{PREPROCESSED_FILE}", header=0)
    full_df['meeting_length'] = full_df['document'].apply(
        lambda x: len(str(x).split()))
    full_df = full_df[full_df['meeting_length'] >= TOPIC_MIN_MEETING_LENGTH]

    get_topics(
        full_df,
        min_doc_freq=TOPIC_MIN_DOC_FREQ,
        max_doc_freq=TOPIC_MAX_DOC_FREQ,
        num_topics=TOPIC_NUM_TOPICS,
        alpha=TOPIC_ALPHA,
        beta=TOPIC_BETA,
        num_top_words=TOPIC_NUM_TOP_WORDS,
        merge_cutoff=TOPIC_MERGE_CUTOFF,
        more_stop_words=TOPIC_MORE_STOP_WORDS,
    )
