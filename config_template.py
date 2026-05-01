# -*- coding: utf-8 -*-
"""
config_template.py
------------------
COPY THIS FILE TO config.py and fill in your local paths.
config.py is git-ignored so individual paths never get committed.

    cp config_template.py config.py
"""

import os

# =============================================================================
# DIRECTORIES  – edit these to match your machine
# =============================================================================

DATA_DIRECTORY   = "/path/to/your/localview_data"
SAVE_DIRECTORY   = "/path/to/your/output_directory"

# =============================================================================
# DATA FILE NAMES  (shared across all scripts — do not rename these)
# =============================================================================

RAW_MEETINGS_TEMPLATE   = "meetings.{year}.parquet"
PREPROCESSED_FILE       = "localview_data_preprocessed.csv"
SEGMENTED_FILE          = "localview_data_segmented.csv"
SENTIMENT_RAW_FILE      = "sentence_sentiment_scores.csv"
SENTIMENT_CLEAN_FILE    = "sentence_sentiment_scores_clean.csv"
SENTIMENT_EXAMPLE_FILE  = "example_sentiment_scores.csv"
SENTIMENT_MEETING_FILE  = "sentiment_meeting_level.csv"
TOPICS_MERGED_TXT       = "merged_topics.txt"
TOPIC_DATA_FILE         = "topic_data_final.csv"
EMBEDDINGS_FILE         = "meeting_embeddings.csv"
SIMILARITY_RAW_FILE     = "meeting_similarity_data.csv"
SIMILARITY_SPLIT_FILE   = "meeting_similarity_data_split.csv"
DYNAMIC_TOPIC_KEYWORDS  = "topic_keywords_top10.txt"
DYNAMIC_TOPIC_NOVELTY   = "meeting_topic_novelty_by_city.csv"
ELECTION_PLACES_FILE    = "election_places.csv"

# =============================================================================
# PREPROCESSING
# =============================================================================

PREPROCESS_YEARS    = list(range(2006, 2024))
MIN_DOCUMENT_LENGTH = 50

# =============================================================================
# SEGMENTER
# =============================================================================

SEGMENT_CHUNK_SIZE    = 1000
SEGMENT_OFFSET_CHUNKS = 0

# =============================================================================
# SENTIMENT
# =============================================================================

SENTIMENT_OFFSET = 0

# =============================================================================
# TOPIC (LDA)
# =============================================================================

TOPIC_MIN_DOC_FREQ       = 0.05
TOPIC_MAX_DOC_FREQ       = 0.95
TOPIC_NUM_TOPICS         = 20
TOPIC_ALPHA              = 0.3845401188473625
TOPIC_BETA               = 0.7896910002727693
TOPIC_MAX_ITER           = 5
TOPIC_NUM_TOP_WORDS      = 20
TOPIC_MERGE_CUTOFF       = 1.0
TOPIC_MORE_STOP_WORDS    = ["uh", "um", "okay", "ok", "yeah"]
TOPIC_MIN_MEETING_LENGTH = 500

# =============================================================================
# SIMILARITY
# =============================================================================

SIMILARITY_STARTING_ROW  = 0
SIMILARITY_TFIDF_MAX_DF  = 0.90
SIMILARITY_TFIDF_MIN_DF  = 2

# =============================================================================
# DYNAMIC TOPICS (BERTopic)
# =============================================================================

DYN_MIN_DOC_FREQ          = 0.05
DYN_MAX_DOC_FREQ          = 0.95
DYN_MIN_MEETING_LENGTH    = 500
DYN_HDBSCAN_MIN_CLUSTER   = 150
DYN_NOVELTY_WINDOW_YEARS  = 2
DYN_TOPIC_THRESHOLD       = 0.01
DYN_MORE_STOP_WORDS       = ["uh", "um", "okay", "ok", "yeah"]
