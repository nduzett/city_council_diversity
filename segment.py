# -*- coding: utf-8 -*-
"""
02_segment.py
-------------
Splits preprocessed meeting documents into sentences using the
WtP SaT sentence segmenter and saves the result to CSV.

Input:  SAVE_DIRECTORY / localview_data_preprocessed.csv
Output: SAVE_DIRECTORY / localview_data_segmented.csv
"""

import time

import pandas as pd
from tqdm import tqdm
from wtpsplit import SaT

from config import (
    SAVE_DIRECTORY,
    PREPROCESSED_FILE, SEGMENTED_FILE,
    SEGMENT_CHUNK_SIZE, SEGMENT_OFFSET_CHUNKS,
)

tqdm.pandas()

# Load segmenter once at module level so it isn't reloaded per chunk.
sat_tokenizer = SaT("sat-3l-sm")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_sentences(input_text: str) -> str:
    """Return pipe-delimited string of sentences from a meeting document."""
    sentence_list = sat_tokenizer.split(input_text)
    return "|".join(sentence_list)


# ---------------------------------------------------------------------------
# Main pipeline step
# ---------------------------------------------------------------------------

def run_segmenter(input_df: pd.DataFrame,
                  chunk_size: int,
                  offset_chunks: int) -> None:
    """
    Segment ``input_df`` in chunks, writing results incrementally to CSV.

    Parameters
    ----------
    input_df      : preprocessed meetings DataFrame
    chunk_size    : number of meetings per processing chunk
    offset_chunks : chunk index to resume from (0 = start from beginning)
    """
    out_path = f"{SAVE_DIRECTORY}\\{SEGMENTED_FILE}"
    start_row = chunk_size * offset_chunks

    for start in range(start_row, len(input_df), chunk_size):
        end   = min(start + chunk_size, len(input_df))
        chunk = input_df.iloc[start:end].copy()
        chunk_num = start // chunk_size

        t0 = time.time()
        print(f"\nSegmenting chunk {chunk_num}  (rows {start}–{end-1})...")
        chunk['sentences'] = chunk['document'].progress_apply(get_sentences)

        elapsed = round((time.time() - t0) / 3600, 2)
        print(f"Chunk {chunk_num} took {elapsed} hours")

        write_header = (start == start_row)
        write_mode   = "w" if write_header else "a"
        chunk.to_csv(out_path, index=False,
                     header=write_header, mode=write_mode)

    print(f"\nSegmented data saved to: {out_path}")


if __name__ == "__main__":
    df = pd.read_csv(f"{SAVE_DIRECTORY}\\{PREPROCESSED_FILE}", header=0)
    run_segmenter(df, SEGMENT_CHUNK_SIZE, SEGMENT_OFFSET_CHUNKS)
