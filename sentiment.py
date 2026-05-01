# -*- coding: utf-8 -*-
"""
03_sentiment.py
---------------
Runs three sentiment models (VADER, TextBlob, RoBERTa) on every sentence
produced by the segmenter, then writes a clean meeting-level summary.

Steps
-----
1. Score every sentence  → sentence_sentiment_scores.csv
2. Clean / parse scores  → sentence_sentiment_scores_clean.csv
3. Collapse to meeting   → sentiment_meeting_level.csv
4. Extract sample rows   → example_sentiment_scores.csv  (optional)

Input:  SAVE_DIRECTORY / localview_data_segmented.csv
Output: see above
"""

import time
from statistics import mean

import numpy as np
import pandas as pd
from scipy.special import softmax
from textblob import TextBlob
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (
    SAVE_DIRECTORY,
    SEGMENTED_FILE,
    SENTIMENT_RAW_FILE, SENTIMENT_CLEAN_FILE,
    SENTIMENT_EXAMPLE_FILE, SENTIMENT_MEETING_FILE,
    SENTIMENT_OFFSET,
)

tqdm.pandas()

# ---------------------------------------------------------------------------
# Load RoBERTa once at module level
# ---------------------------------------------------------------------------
_ROBERTA_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"
roberta_tokenizer = AutoTokenizer.from_pretrained(_ROBERTA_PATH)
roberta_config    = AutoConfig.from_pretrained(_ROBERTA_PATH)
roberta_model     = AutoModelForSequenceClassification.from_pretrained(_ROBERTA_PATH)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def vader_sentiment_scores(text: str) -> list:
    sid = SentimentIntensityAnalyzer()
    d   = sid.polarity_scores(text)
    return [d['neg'] * 100, d['neu'] * 100, d['pos'] * 100]


def textblob_sentiment_scores(text: str) -> list:
    tb = TextBlob(text).sentiment
    return [tb.polarity, tb.subjectivity]


def roberta_sentiment_scores(text: str) -> list:
    try:
        enc    = roberta_tokenizer(text, return_tensors='pt')
        output = roberta_model(**enc)
        scores = softmax(output[0][0].detach().numpy())
        return [scores[0] * 100, scores[1] * 100, scores[2] * 100]
    except Exception:
        return [np.nan, np.nan, np.nan]


# ---------------------------------------------------------------------------
# Classification helpers  (used in clean_sentiment step)
# ---------------------------------------------------------------------------

def classify_neg_vd(row):
    return int(row['vader_neg'] > 33 and row['vader_neg'] > row['vader_neu'] and row['vader_neg'] > row['vader_pos'])

def classify_neu_vd(row):
    return int(row['vader_neu'] > 33 and row['vader_neu'] > row['vader_neg'] and row['vader_neu'] > row['vader_pos'])

def classify_pos_vd(row):
    return int(row['vader_pos'] > 33 and row['vader_pos'] > row['vader_neu'] and row['vader_pos'] > row['vader_neg'])

def classify_neg_tb(row):
    return int(row['tb_polarity'] <= -33)

def classify_neu_tb(row):
    return int(-33 < row['tb_polarity'] < 33)

def classify_pos_tb(row):
    return int(row['tb_polarity'] >= 33)

def classify_neg_rob(row):
    return int(row['rob_neg'] > 33 and row['rob_neg'] > row['rob_neu'] and row['rob_neg'] > row['rob_pos'])

def classify_neu_rob(row):
    return int(row['rob_neu'] > 33 and row['rob_neu'] > row['rob_neg'] and row['rob_neu'] > row['rob_pos'])

def classify_pos_rob(row):
    return int(row['rob_pos'] > 33 and row['rob_pos'] > row['rob_neu'] and row['rob_pos'] > row['rob_neg'])


# ---------------------------------------------------------------------------
# Main pipeline steps
# ---------------------------------------------------------------------------

def get_sentiment(df: pd.DataFrame, offset: int = 0) -> None:
    """
    Score every sentence in every meeting and write results incrementally.

    Parameters
    ----------
    df     : segmented meetings DataFrame (must contain 'sentences' column)
    offset : meeting index to resume from
    """
    out_path    = f"{SAVE_DIRECTORY}\\{SENTIMENT_RAW_FILE}"
    num_meetings = len(df)
    times        = []
    j            = offset

    while j < num_meetings:
        t0 = time.time()
        if j != offset:
            remaining = round(mean(times) * (num_meetings - j) / 3600, 2)
            print(f"{round(100 * j / num_meetings, 2)}% complete — "
                  f"{j} meetings done, ~{remaining} hours remaining")

        sentences    = df['sentences'][j].split("|")
        meeting_rows = []

        for sentence in sentences:
            scores = []
            scores.extend(vader_sentiment_scores(sentence))
            scores.extend(textblob_sentiment_scores(sentence))
            scores.extend(roberta_sentiment_scores(sentence))
            meeting_rows.append([
                df['meeting_id'][j], df['state_name'][j],
                df['place_name'][j],  df['meeting_date'][j],
                sentence, "|".join(map(str, scores))
            ])

        try:
            save_df = pd.DataFrame(
                meeting_rows,
                columns=['meeting_id', 'state', 'city',
                         'meeting_date', 'sentence', 'scores'])
            write_header = (j == 0)
            write_mode   = "w" if write_header else "a"
            save_df.to_csv(out_path, index=False,
                           header=write_header, mode=write_mode)
        except Exception as e:
            print(f"Write error at meeting {j}: {e}")

        times.append(time.time() - t0)
        j += 1

    print(f"\nRaw sentiment scores saved to: {out_path}")


def clean_sentiment() -> None:
    """Parse raw scores, classify sentences, collapse to meeting level."""
    raw_path   = f"{SAVE_DIRECTORY}\\{SENTIMENT_RAW_FILE}"
    clean_path = f"{SAVE_DIRECTORY}\\{SENTIMENT_CLEAN_FILE}"
    meet_path  = f"{SAVE_DIRECTORY}\\{SENTIMENT_MEETING_FILE}"

    df = pd.read_csv(raw_path, header=0)

    # -- Parse pipe-delimited scores column ----------------------------------
    print("Splitting scores column...")
    score_cols = df['scores'].str.split('|', expand=True)
    score_cols.rename(columns={
        0: 'vader_neg', 1: 'vader_neu', 2: 'vader_pos',
        3: 'tb_polarity', 4: 'tb_subjectivity',
        5: 'rob_neg', 6: 'rob_neu', 7: 'rob_pos'
    }, inplace=True)
    df = pd.concat([df, score_cols], axis=1)

    print("Converting score columns to float...")
    for col in ['vader_neg', 'vader_neu', 'vader_pos',
                'tb_polarity', 'tb_subjectivity',
                'rob_neg', 'rob_neu', 'rob_pos']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float64)

    # -- Classify sentences --------------------------------------------------
    print("Classifying sentences...")
    print("  vader...")
    df['vd_negative'] = df.progress_apply(classify_neg_vd, axis=1)
    df['vd_neutral']  = df.progress_apply(classify_neu_vd, axis=1)
    df['vd_positive'] = df.progress_apply(classify_pos_vd, axis=1)
    print("  textblob...")
    df['tb_polarity']  = 100 * df['tb_polarity']
    df['tb_negative']  = df.progress_apply(classify_neg_tb, axis=1)
    df['tb_neutral']   = df.progress_apply(classify_neu_tb, axis=1)
    df['tb_positive']  = df.progress_apply(classify_pos_tb, axis=1)
    print("  roberta...")
    df['rob_negative'] = df.progress_apply(classify_neg_rob, axis=1)
    df['rob_neutral']  = df.progress_apply(classify_neu_rob, axis=1)
    df['rob_positive'] = df.progress_apply(classify_pos_rob, axis=1)

    df.to_csv(clean_path, index=False, header=True)
    print(f"Clean sentence scores saved to: {clean_path}")

    # -- Collapse to meeting level -------------------------------------------
    print("Collapsing to meeting level...")
    keep_cols = ['meeting_id', 'state', 'city', 'meeting_date', 'sentence',
                 'vd_negative', 'vd_neutral', 'vd_positive',
                 'tb_negative', 'tb_neutral', 'tb_positive',
                 'rob_negative', 'rob_neutral', 'rob_positive',
                 'tb_subjectivity']
    meeting_df = df[keep_cols]

    collapsed = meeting_df.groupby('meeting_id', as_index=False).agg({
        'state': 'first', 'city': 'first',
        'meeting_date': 'first', 'sentence': 'first',
        'vd_negative': 'mean', 'vd_neutral': 'mean', 'vd_positive': 'mean',
        'tb_negative': 'mean', 'tb_neutral': 'mean', 'tb_positive': 'mean',
        'rob_negative': 'mean', 'rob_neutral': 'mean', 'rob_positive': 'mean',
        'tb_subjectivity': 'mean'
    })
    collapsed['merge_string'] = collapsed['sentence'].apply(lambda x: x[:50])

    collapsed.to_csv(meet_path, index=False, header=True)
    print(f"Meeting-level sentiment saved to: {meet_path}")


def extract_sample(n: int = 5000, random_state: int = 42) -> None:
    """Save a random sample of clean sentences for inspection / validation."""
    in_path  = f"{SAVE_DIRECTORY}\\{SENTIMENT_CLEAN_FILE}"
    out_path = f"{SAVE_DIRECTORY}\\{SENTIMENT_EXAMPLE_FILE}"
    df = pd.read_csv(in_path, header=0)
    df.sample(n=n, random_state=random_state).to_csv(out_path, index=False, header=True)
    print(f"Example sentences saved to: {out_path}")


if __name__ == "__main__":
    master_df = pd.read_csv(
        f"{SAVE_DIRECTORY}\\{SEGMENTED_FILE}", header=0)
    get_sentiment(master_df, SENTIMENT_OFFSET)
    clean_sentiment()
    extract_sample()
