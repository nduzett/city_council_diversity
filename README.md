# LocalView NLP Pipeline

A modular Python pipeline for large-scale text analysis of LocalView municipal
meeting transcripts. The pipeline preprocesses raw transcripts, segments them
into sentences, scores sentiment at the sentence level, models topics with LDA
and BERTopic, and computes textual similarity between meetings over time.

---

## Data sources and acknowledgements

This pipeline is built on two publicly available datasets. We are grateful
to the teams behind both projects for making their work openly available.

### LocalView

Transcript data comes from the **LocalView** project, which provides a
large-scale collection of recorded and transcribed U.S. local government
meetings.

- Project site: [localview.net](https://www.localview.net/)
- Data repository: [Harvard Dataverse — doi:10.7910/DVN/NJTBEM](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NJTBEM)

We thank the LocalView team for assembling and maintaining this dataset and
for making it available to the research community.

### American Local Government Elections Database (ALGED)

Election and municipality matching data used in the similarity pipeline
comes from the **American Local Government Elections Database**.

- Publication: [*Scientific Data* — doi:10.1038/s41597-023-02792-x](https://www.nature.com/articles/s41597-023-02792-x)

We thank the authors of ALGED for compiling and sharing this resource.

---

> **Runtime note:** The segmenting and sentiment steps are the most
> computationally intensive parts of this pipeline. Depending on hardware,
> the segmenter can take **tens to several hundred hours** to process the full
> dataset. The sentiment step, which runs three models (VADER, TextBlob, and
> RoBERTa) on every sentence of every meeting, is similarly demanding and
> should be expected to run for a **comparable or longer duration**. Both
> steps write output incrementally so they can be safely interrupted and
> resumed — see the [Resuming interrupted runs](#resuming-interrupted-runs)
> section below.

---

## Repository structure

```
localview_nlp/
├── config_template.py     # Template — copy to config.py and fill in your paths
├── run_pipeline.py        # Master script — toggle individual steps on/off here
│
├── preprocess.py          # Step 1 — clean raw parquet files
├── segment.py             # Step 2 — split documents into sentences
├── sentiment.py           # Step 3 — score sentences + clean + collapse
├── topics.py              # Step 4 — LDA topic modeling
├── similarity.py          # Step 5 — TF-IDF embeddings + cosine similarity
├── dynamic_topics.py      # Step 6 — BERTopic + per-city novelty scoring
│
├── requirements.txt       # Python dependencies
├── CHANGELOG.md           # Parameter decisions and version history
└── README.md
```

---

## File reference

Files are listed in the order they should be run.

---

### `config_template.py` / `config.py`

**Purpose:** Single source of truth for all file paths, output filenames, and
tunable parameters. Every other script imports its constants from here, so
changing a filename or parameter in one place propagates automatically to all
scripts that use it.

`config.py` is listed in `.gitignore` and is never committed. Each team member
copies `config_template.py` to `config.py` and sets their own local paths.

**Dependencies:** `os` (stdlib only)

---

### `run_pipeline.py`

**Purpose:** Master orchestration script. Contains a set of boolean flags at
the top of the file — set a flag to `True` to run that step, `False` to skip
it. Steps are executed in the same order they appeared in the original
monolithic script. Heavy modules are imported lazily (only when their step is
enabled) to keep startup time low when running only one or two steps.

**Dependencies:** `config.py`; imports from each step module on demand

---

### `preprocess.py` — Step 1

**Purpose:** Loads raw LocalView parquet files for all configured years, filters
to municipal council meetings, and applies a full text-cleaning pipeline:

1. Remove bracketed pause annotations (e.g. `[laughter]`, `(inaudible)`)
2. Lowercase all text
3. Remove punctuation
4. Remove `<no caption available>` placeholder strings
5. Strip proper nouns via spaCy POS tagging
6. Lemmatize tokens using NLTK WordNetLemmatizer with POS-aware mapping

Meetings shorter than the minimum document length (default: 50 characters) are
dropped. Output is a flat CSV with one row per meeting.

**Input:** `DATA_DIRECTORY/meetings.{year}.parquet` for each year in `PREPROCESS_YEARS`

**Output:** `SAVE_DIRECTORY/localview_data_preprocessed.csv`

**Dependencies:** `pandas`, `nltk` (punkt_tab, averaged_perceptron_tagger_eng,
wordnet, stopwords), `spacy` (en_core_web_sm), `tqdm`, `re`, `time`

---

### `segment.py` — Step 2

**Purpose:** Splits each preprocessed meeting document into individual sentences
using the WtP SaT (`sat-3l-sm`) neural sentence segmenter. Sentences are stored
as a single pipe-delimited string per meeting row. Processing runs in configurable
chunks and writes output incrementally to CSV so that long runs can be safely
resumed.

> ⏱ **Runtime warning:** This step can take **tens to several hundred hours**
> on the full dataset. Set `SEGMENT_OFFSET_CHUNKS` in `config.py` to resume
> from a specific chunk after an interruption.

**Input:** `SAVE_DIRECTORY/localview_data_preprocessed.csv`

**Output:** `SAVE_DIRECTORY/localview_data_segmented.csv`

**Dependencies:** `pandas`, `wtpsplit`, `tqdm`, `time`

---

### `sentiment.py` — Step 3

**Purpose:** Scores every sentence in every meeting using three sentiment models,
then cleans and collapses the results to the meeting level. Contains three
callable sub-steps:

- **`get_sentiment()`** — runs VADER, TextBlob, and RoBERTa
  (`cardiffnlp/twitter-roberta-base-sentiment-latest`) on each sentence; scores
  are stored as a pipe-delimited string and written incrementally to CSV.
- **`clean_sentiment()`** — parses the raw scores string into named float
  columns; applies majority-class classification rules (>33% and highest of the
  three classes) to assign binary negative/neutral/positive labels for each
  model; collapses to meeting-level means.
- **`extract_sample()`** — draws a random sample of 5,000 clean sentences for
  manual inspection and validation.

> ⏱ **Runtime warning:** Running three models sentence-by-sentence across the
> full corpus is among the most time-intensive steps in the pipeline. Expect
> **comparable or longer runtime than the segmenter**, potentially reaching
> several hundred hours. Set `SENTIMENT_OFFSET` in `config.py` to resume from
> a specific meeting index.

**Input:** `SAVE_DIRECTORY/localview_data_segmented.csv`

**Output:**
- `SAVE_DIRECTORY/sentence_sentiment_scores.csv` (raw)
- `SAVE_DIRECTORY/sentence_sentiment_scores_clean.csv` (parsed + classified)
- `SAVE_DIRECTORY/sentiment_meeting_level.csv` (collapsed means)
- `SAVE_DIRECTORY/example_sentiment_scores.csv` (5,000-row sample)

**Dependencies:** `pandas`, `numpy`, `scipy` (softmax), `textblob`,
`vaderSentiment`, `transformers` (AutoTokenizer, AutoModelForSequenceClassification),
`torch`, `tqdm`, `time`, `statistics`

---

### `topics.py` — Step 4

**Purpose:** Fits an LDA topic model on the preprocessed meeting documents,
merges topics whose cosine similarity exceeds a configurable threshold, prints
and saves the top keywords for each merged topic, and saves per-document
topic probability distributions.

Key internal functions:

- **`merge_topics()`** — computes cosine similarity between all topic-word
  distributions and groups topics above the merge threshold into clusters.
- **`merge_topic_distributions()`** — sums the document-level probabilities
  of all topics within a merged group.
- **`print_merged_topics()`** — writes the top-N keywords for each merged
  topic to both stdout and a text file.
- **`get_topics()`** — main entry point; handles vectorization, model fitting,
  merging, and saving.

**Input:** `SAVE_DIRECTORY/localview_data_preprocessed.csv`

**Output:**
- `SAVE_DIRECTORY/topic_data_final.csv` (per-document topic probabilities)
- `SAVE_DIRECTORY/merged_topics.txt` (top keywords per merged topic)

**Dependencies:** `pandas`, `numpy`, `nltk` (stopwords), `scikit-learn`
(TfidfVectorizer, LatentDirichletAllocation), `gc`, `time`

---

### `similarity.py` — Step 5

**Purpose:** Computes TF-IDF-based cosine similarity between each meeting and
all prior meetings in the same city/state, enabling longitudinal analysis of
how meeting content changes over time. Contains three callable sub-steps:

- **`run_embedding()`** — merges the preprocessed data against the
  election-places lookup, computes per-meeting TF-IDF vectors, and saves them
  to CSV.
- **`run_similarity()`** — for each meeting (sorted by city/state/date),
  computes cosine similarity against all earlier meetings in the same
  city/state. Results are written row-by-row so the step is resumable.
- **`clean_similarity()`** — explodes the pipe-delimited similarity scores and
  dates into a long-format CSV with one score per row.

Also contains `merge_election_places()`, a helper that inner-joins the meeting
data to an election-places lookup table, normalizing state names and place
name suffixes for a reliable merge.

**Input:** `SAVE_DIRECTORY/localview_data_preprocessed.csv`,
`DATA_DIRECTORY/election_places.csv`

**Output:**
- `SAVE_DIRECTORY/meeting_embeddings.csv`
- `SAVE_DIRECTORY/meeting_similarity_data.csv`
- `SAVE_DIRECTORY/meeting_similarity_data_split.csv`

**Dependencies:** `pandas`, `numpy`, `scikit-learn` (TfidfVectorizer,
cosine_similarity), `gc`, `tqdm`

---

### `dynamic_topics.py` — Step 6

**Purpose:** Fits a BERTopic model (backed by UMAP dimensionality reduction and
HDBSCAN clustering) on the preprocessed meeting documents, then computes two
novelty measures for each meeting within its city-state:

- **Novelty score** — cosine distance between a meeting's topic distribution
  and the mean distribution of all meetings in the same city-state over the
  prior two years. Higher values indicate a more unusual meeting.
- **New topic count** — the number of topics active in a meeting (probability
  above threshold) that did not appear in the same city-state in the prior two
  years.

**Input:** `SAVE_DIRECTORY/localview_data_preprocessed.csv`

**Output:**
- `SAVE_DIRECTORY/topic_keywords_top10.txt` (top 10 words per BERTopic topic)
- `SAVE_DIRECTORY/meeting_topic_novelty_by_city.csv`

**Dependencies:** `pandas`, `numpy`, `nltk` (stopwords), `scikit-learn`
(TfidfVectorizer, cosine_similarity), `bertopic`, `hdbscan`, `umap-learn`, `os`

---

### `requirements.txt`

**Purpose:** Pinned list of all third-party Python packages required by the
pipeline. Install with `pip install -r requirements.txt`. Includes a reminder
to run `python -m spacy download en_core_web_sm` separately after installation.

---

## Quick start

### 1 — Clone and create a virtual environment

```bash
git clone https://github.com/your-org/localview-nlp.git
cd localview-nlp
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2 — Configure local paths

```bash
cp config_template.py config.py
# Open config.py and set DATA_DIRECTORY and SAVE_DIRECTORY
```

`config.py` is git-ignored — your local paths are never committed.

### 3 — Run the pipeline

Open `run_pipeline.py`, set the step flags you want to `True`, then:

```bash
python run_pipeline.py
```

Individual steps can also be run directly:

```bash
python preprocess.py
python segment.py
# etc.
```

---

## Pipeline data flow

```
meetings.{year}.parquet  ← raw input in DATA_DIRECTORY
        │
        ▼ preprocess.py
localview_data_preprocessed.csv
        │
        ├──▶ segment.py ──────────────────▶ localview_data_segmented.csv
        │                                           │
        │                                           ▼ sentiment.py
        │                               sentence_sentiment_scores.csv
        │                               sentence_sentiment_scores_clean.csv
        │                               sentiment_meeting_level.csv
        │                               example_sentiment_scores.csv
        │
        ├──▶ topics.py ───────────────────▶ topic_data_final.csv
        │                                   merged_topics.txt
        │
        ├──▶ similarity.py ───────────────▶ meeting_embeddings.csv
        │       ↑ also reads                meeting_similarity_data.csv
        │       election_places.csv         meeting_similarity_data_split.csv
        │
        └──▶ dynamic_topics.py ───────────▶ topic_keywords_top10.txt
                                            meeting_topic_novelty_by_city.csv
```

---

## Resuming interrupted runs

The three longest-running steps write output incrementally and can be resumed
without re-processing already-completed work. Adjust the relevant offset in
`config.py` before restarting:

| Step | Config variable | Unit |
|---|---|---|
| Segmenter | `SEGMENT_OFFSET_CHUNKS` | chunk index (chunk size × offset = first meeting row) |
| Sentiment scoring | `SENTIMENT_OFFSET` | meeting row index |
| Similarity computation | `SIMILARITY_STARTING_ROW` | meeting row index |

---

## Team workflow recommendations

- **Never commit data or outputs.** The `.gitignore` already excludes all CSVs
  and parquets. Use shared cloud storage (S3, Google Drive, etc.) for data and
  document the expected folder structure in a shared note.
- **Branch per experiment.** Name branches after what is being tested
  (e.g., `experiment/bertopic-min-cluster-200`) rather than by person.
- **Tag stable runs.** When a full pipeline run produces results going into a
  paper or report, tag the commit: `git tag v1.0-paper-results`.
- **Log parameter decisions in `CHANGELOG.md`.** All tuning rationale belongs
  there, not in Slack threads.
- **Use GitHub Issues for open methodological questions** so the whole team
  can track unresolved decisions alongside the code.
