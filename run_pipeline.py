# -*- coding: utf-8 -*-
"""
run_pipeline.py
---------------
Master script for the LocalView NLP pipeline.

Toggle each step on/off with the flags below, then run:

    python run_pipeline.py

Steps run in the order they appear in the original monolithic script:
  1. preprocess       – clean raw parquet files
  2. segment          – split documents into sentences
  3. sentiment        – score sentences (VADER / TextBlob / RoBERTa)
  4. clean_sentiment  – parse scores, classify, collapse to meeting level
  5. extract_sample   – save a random sample of clean sentences (optional)
  6. topics           – fit LDA model, merge topics
  7. embedding        – build TF-IDF vectors per meeting
  8. similarity       – compute cosine similarity between meetings
  9. clean_similarity – explode similarity output to long format
 10. dynamic_topics   – fit BERTopic, compute novelty scores
"""

# =============================================================================
# STEP FLAGS  –  set to True to run, False to skip
# =============================================================================

RUN_PREPROCESS        = False
RUN_SEGMENT           = False
RUN_SENTIMENT         = False
RUN_CLEAN_SENTIMENT   = False
RUN_EXTRACT_SAMPLE    = False
RUN_TOPICS            = False
RUN_EMBEDDING         = False
RUN_SIMILARITY        = False
RUN_CLEAN_SIMILARITY  = False
RUN_DYNAMIC_TOPICS    = False

# =============================================================================
# IMPORTS  (only load heavy modules if the corresponding step is enabled)
# =============================================================================

import time

from config import SAVE_DIRECTORY  # confirm config is importable before anything else

_pipeline_start = time.time()


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# 1. PREPROCESSING
# =============================================================================
if RUN_PREPROCESS:
    _banner("Step 1 – Preprocessing")
    from preprocess import run_preprocessing
    run_preprocessing()

# =============================================================================
# 2. SEGMENTING
# =============================================================================
if RUN_SEGMENT:
    _banner("Step 2 – Segmenting")
    import pandas as pd
    from config import PREPROCESSED_FILE, SEGMENT_CHUNK_SIZE, SEGMENT_OFFSET_CHUNKS
    from segment import run_segmenter
    df = pd.read_csv(f"{SAVE_DIRECTORY}\\{PREPROCESSED_FILE}", header=0)
    run_segmenter(df, SEGMENT_CHUNK_SIZE, SEGMENT_OFFSET_CHUNKS)

# =============================================================================
# 3. SENTIMENT SCORING
# =============================================================================
if RUN_SENTIMENT:
    _banner("Step 3 – Sentiment Scoring")
    import pandas as pd
    from config import SEGMENTED_FILE, SENTIMENT_OFFSET
    from sentiment import get_sentiment
    master_df = pd.read_csv(f"{SAVE_DIRECTORY}\\{SEGMENTED_FILE}", header=0)
    get_sentiment(master_df, SENTIMENT_OFFSET)

# =============================================================================
# 4. CLEAN SENTIMENT
# =============================================================================
if RUN_CLEAN_SENTIMENT:
    _banner("Step 4 – Cleaning Sentiment")
    from sentiment import clean_sentiment
    clean_sentiment()

# =============================================================================
# 5. EXTRACT SAMPLE SENTENCES  (optional validation aid)
# =============================================================================
if RUN_EXTRACT_SAMPLE:
    _banner("Step 5 – Extracting Sample Sentences")
    from sentiment import extract_sample
    extract_sample()

# =============================================================================
# 6. TOPIC MODELING (LDA)
# =============================================================================
if RUN_TOPICS:
    _banner("Step 6 – Topic Modeling (LDA)")
    import pandas as pd
    from config import (
        PREPROCESSED_FILE,
        TOPIC_MIN_DOC_FREQ, TOPIC_MAX_DOC_FREQ,
        TOPIC_NUM_TOPICS, TOPIC_ALPHA, TOPIC_BETA,
        TOPIC_NUM_TOP_WORDS, TOPIC_MERGE_CUTOFF,
        TOPIC_MORE_STOP_WORDS, TOPIC_MIN_MEETING_LENGTH,
    )
    from topics import get_topics

    full_df = pd.read_csv(f"{SAVE_DIRECTORY}\\{PREPROCESSED_FILE}", header=0)
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

# =============================================================================
# 7. EMBEDDING (TF-IDF vectors for similarity)
# =============================================================================
if RUN_EMBEDDING:
    _banner("Step 7 – Building TF-IDF Embeddings")
    from similarity import run_embedding
    run_embedding()

# =============================================================================
# 8. SIMILARITY COMPUTATION
# =============================================================================
if RUN_SIMILARITY:
    _banner("Step 8 – Computing Meeting Similarity")
    from config import SIMILARITY_STARTING_ROW
    from similarity import run_similarity
    run_similarity(SIMILARITY_STARTING_ROW)

# =============================================================================
# 9. CLEAN SIMILARITY OUTPUT
# =============================================================================
if RUN_CLEAN_SIMILARITY:
    _banner("Step 9 – Cleaning Similarity Output")
    from similarity import clean_similarity
    clean_similarity()

# =============================================================================
# 10. DYNAMIC TOPICS (BERTopic + novelty scoring)
# =============================================================================
if RUN_DYNAMIC_TOPICS:
    _banner("Step 10 – Dynamic Topics (BERTopic)")
    from dynamic_topics import run_dynamic_topics
    run_dynamic_topics()

# =============================================================================
# DONE
# =============================================================================
_elapsed = round((time.time() - _pipeline_start) / 3600, 2)
print(f"\nPipeline complete in {_elapsed} hours.")
