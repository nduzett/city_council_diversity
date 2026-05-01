# -*- coding: utf-8 -*-
"""
config.py
---------
Shared configuration for the LocalView NLP pipeline.
Edit the directories and parameters here before running.
"""

import os

# =============================================================================
# DIRECTORIES
# Edit these paths to match your local environment.
# =============================================================================

DATA_DIRECTORY   = "C:\\Users\\Neil\\Desktop\\ccd_nlp_master\\localview_data"
SAVE_DIRECTORY   = "C:\\Users\\Neil\\Desktop\\ccd_nlp_master"

# =============================================================================
# DATA FILE NAMES  (relative to SAVE_DIRECTORY unless noted)
# All scripts import these constants so filenames stay in sync.
# =============================================================================

# Input raw data
RAW_MEETINGS_TEMPLATE   = "meetings.{year}.parquet"        # lives in DATA_DIRECTORY

# Preprocessing output
PREPROCESSED_FILE       = "localview_data_preprocessed.csv"

# Segmenter output
SEGMENTED_FILE          = "localview_data_segmented.csv"

# Sentiment outputs
SENTIMENT_RAW_FILE      = "sentence_sentiment_scores.csv"
SENTIMENT_CLEAN_FILE    = "sentence_sentiment_scores_clean.csv"
SENTIMENT_EXAMPLE_FILE  = "example_sentiment_scores.csv"
SENTIMENT_MEETING_FILE  = "sentiment_meeting_level.csv"

# Topic outputs
TOPICS_MERGED_TXT       = "merged_topics.txt"
TOPIC_DATA_FILE         = "topic_data_final.csv"

# Similarity / embedding outputs
EMBEDDINGS_FILE         = "meeting_embeddings.csv"
SIMILARITY_RAW_FILE     = "meeting_similarity_data.csv"
SIMILARITY_SPLIT_FILE   = "meeting_similarity_data_split.csv"

# Dynamic topics outputs
DYNAMIC_TOPIC_KEYWORDS  = "topic_keywords_top10.txt"
DYNAMIC_TOPIC_NOVELTY   = "meeting_topic_novelty_by_city.csv"

# Election places lookup (lives in DATA_DIRECTORY)
ELECTION_PLACES_FILE    = "election_places.csv"

# =============================================================================
# PREPROCESSING PARAMETERS
# =============================================================================

PREPROCESS_YEARS = list(range(2006, 2024))
MIN_DOCUMENT_LENGTH = 50          # characters; shorter meetings are dropped

# =============================================================================
# SEGMENTER PARAMETERS
# =============================================================================

SEGMENT_CHUNK_SIZE   = 1000
SEGMENT_OFFSET_CHUNKS = 0         # chunk index to start from (for resuming)

# =============================================================================
# SENTIMENT PARAMETERS
# =============================================================================

SENTIMENT_OFFSET = 0              # meeting index to start from (for resuming)

# =============================================================================
# TOPIC (LDA) PARAMETERS
# =============================================================================

TOPIC_MIN_DOC_FREQ  = 0.05
TOPIC_MAX_DOC_FREQ  = 0.95
TOPIC_NUM_TOPICS    = 20
TOPIC_ALPHA         = 0.3845401188473625
TOPIC_BETA          = 0.7896910002727693
TOPIC_MAX_ITER      = 5
TOPIC_NUM_TOP_WORDS = 20
TOPIC_MERGE_CUTOFF  = 1.0
TOPIC_MORE_STOP_WORDS = ["uh", "um", "okay", "ok", "yeah"]
TOPIC_MIN_MEETING_LENGTH = 500    # words; shorter meetings are dropped

# =============================================================================
# SIMILARITY PARAMETERS
# =============================================================================

SIMILARITY_STARTING_ROW = 0      # row index to start from (for resuming)
SIMILARITY_TFIDF_MAX_DF = 0.90
SIMILARITY_TFIDF_MIN_DF = 2

# =============================================================================
# DYNAMIC TOPICS (BERTopic) PARAMETERS
# =============================================================================

DYN_MIN_DOC_FREQ          = 0.05
DYN_MAX_DOC_FREQ          = 0.95
DYN_MIN_MEETING_LENGTH    = 500   # words
DYN_HDBSCAN_MIN_CLUSTER   = 150
DYN_NOVELTY_WINDOW_YEARS  = 2
DYN_TOPIC_THRESHOLD       = 0.01
DYN_MORE_STOP_WORDS       = ["uh", "um", "okay", "ok", "yeah"]
