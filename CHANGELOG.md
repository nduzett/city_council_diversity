# Changelog

All parameter decisions, model choices, and notable pipeline changes are
recorded here. Entries are listed newest-first within each version.

---

## [v1.0] — Modular refactor (current)

This version reorganizes the original monolithic script
(`localview_nlp_v5_master.py`) into separate module files. No parameter
values were changed during the refactor. All decisions below were made
during development of the original script and are carried forward as-is.

---

## Parameter decisions (from original script)

### Preprocessing

**Years included: 2006–2023**
All available LocalView years are loaded and concatenated before cleaning.
Only meetings with `place_govt == 'MUNICIPAL COUNCIL'` are retained.

**Minimum document length: 50 characters**
Meetings shorter than 50 characters after preprocessing are dropped as
effectively empty. This threshold is deliberately permissive — it is
intended only to remove records that contain nothing usable, not to filter
short-but-valid meetings.

**Proper noun removal**
Proper nouns are stripped using spaCy (`en_core_web_sm`) POS tagging before
lemmatization. The motivation is to prevent place names, personal names, and
organization names from dominating topic and similarity signals. Documents
over 2,000,000 characters are too large for spaCy's default limits; these
are returned as a single space rather than processed. Documents between
1,000,000 and 2,000,000 characters are split and processed in two halves.

---

### Segmenter

**Model: `sat-3l-sm` (WtP SaT)**
The small 3-layer SaT model was chosen as a balance between segmentation
quality and processing speed over a very large corpus. A larger model would
improve sentence boundary accuracy but is not tractable at this scale.

**Chunk size: 1,000 meetings**
Output is written to CSV after every 1,000 meetings so that progress is
preserved in the event of a crash or interruption.

---

### Sentiment

**Models: VADER + TextBlob + RoBERTa (ensemble)**
Three models are run in parallel to enable downstream comparison of
rule-based (VADER, TextBlob) versus neural (RoBERTa) approaches.
`cardiffnlp/twitter-roberta-base-sentiment-latest` was selected because it
was trained on informal short text, which is closer in register to spoken
municipal transcripts than models trained on formal corpora.

**RoBERTa error handling**
A broad `except` block is used around the RoBERTa call so that a single
sentence that causes a tokenizer or inference error does not halt the entire
run. Failed sentences receive `NaN` scores.

**Sentence classification threshold: 33%**
A sentence is labeled as belonging to a sentiment class if that class score
exceeds 33% AND is the highest of the three class scores. This avoids
assigning a majority label when scores are nearly equal (e.g. 34/33/33).

**Meeting-level aggregation: mean**
Sentence-level binary classifications are averaged to the meeting level,
yielding a continuous "proportion of sentences classified as X" measure
for each model and each sentiment class.

---

### Topic model (LDA)

#### Vectorization

**TF-IDF vectorizer (not raw counts)**
TF-IDF was used rather than raw term counts to down-weight words that
appear in nearly every meeting and therefore carry little discriminating
information.

**Minimum document frequency: 0.05 (5%)**
Terms appearing in fewer than 5% of meetings are excluded. These are
typically highly idiosyncratic terms (place-specific proper nouns that
survived preprocessing, transcription errors) that would otherwise fragment
topics.

**Maximum document frequency: 0.95 (95%)**
Terms appearing in more than 95% of meetings are excluded. These are
effectively stopwords for this specific corpus — words so common in
municipal meetings that they do not differentiate topics.

**Minimum meeting length filter: 500 words**
Meetings shorter than 500 words are excluded before fitting the topic model.
Very short meetings provide too little signal for reliable topic assignment
and can distort the model's learned distributions.

#### Stop words

The pipeline uses NLTK's standard English stop word list plus a
**custom extension** of filler words specific to transcribed speech.
The custom list has evolved through three iterations:

**Iteration 1 — Broad filler + place-name stopwords**
```python
["nt", "uh", "um", "000", "know", "get", "would", "city", "think",
 "like", "say", "thank", "want", "yes", "well", "come", "year",
 "make", "right", "sioux", "dakota", "ah", "oh", "good", "ok", "eight"]
```
Rationale: removed spoken-language filler words (`uh`, `um`, `nt` for
"don't"/"can't" artifacts) and high-frequency but semantically vacuous
conversational words (`know`, `get`, `think`, `like`). Also removed
`sioux` and `dakota` as location-specific proper nouns that survived
the preprocessing step and were appearing across topics.

**Iteration 2 — Expanded to include more conversational tokens**
```python
["nt", "uh", "um", "000", "know", "get", "would", "city", "think",
 "like", "say", "thank", "want", "yes", "well", "come", "year",
 "make", "right", "sioux", "dakota", "ah", "oh", "good", "ok",
 "eight", "hello", "today", "thing", "okay", "time", "look", "work",
 "see", "people", "need", "council", "question", "take", "really",
 "yeah", "two", "also", "could", "back", "talk", "na", "mean", "gon"]
```
Rationale: additional review of topic keywords revealed that high-frequency
meeting procedural language (`council`, `question`) and common conversational
tokens (`people`, `need`, `work`, `talk`) were dominating several topics
without adding interpretive value. `na` and `gon` are transcription
artifacts from "gonna" / "wanna" splitting.

**Iteration 3 — Minimal list (current)**
```python
["uh", "um", "okay", "ok", "yeah"]
```
Rationale: the more aggressive stop word lists in iterations 1 and 2 were
removing words with genuine substantive meaning in context (e.g. `work`,
`people`, `need`, `council`). The decision was made to rely on the TF-IDF
`min_df`/`max_df` thresholds to handle corpus-specific high-frequency words,
and to keep the custom stop word list to only the clearest speech-artifact
tokens with no semantic content.

#### LDA hyperparameters

**Number of topics: 20**
*(Note: the main configuration block in the original file set `num_topics = 15`
at the call site, but the model was hardcoded to `n_components=20` based on
the results of an earlier randomized search. The value of 20 reflects the
search result and is the value actually used. The `15` entry in the
configuration block appears to be a stale value that was not propagated.
This is tracked as a known inconsistency to resolve.)*

**Alpha (doc_topic_prior): 0.3845401188473625**
This value was selected by `RandomizedSearchCV` over the range `uniform(0.01, 1.0)`,
using log-likelihood as the scoring criterion, with 20 random trials and
3-fold cross-validation. The search was run once over the full corpus and the
best-found value was then hardcoded to avoid re-running the expensive search
on subsequent iterations.

Alpha controls the document-topic distribution sparsity. Lower values
concentrate each document's probability mass on fewer topics (sparser
per-document distributions); higher values spread it across more topics.
A value of ~0.38 reflects moderately concentrated topic distributions,
consistent with the expectation that most municipal meetings focus on a
small number of policy areas at a time.
*(For reference: a common rule-of-thumb starting value is `1 / num_topics`,
which for 20 topics would be 0.05.)*

**Beta (topic_word_prior): 0.7896910002727693**
Also selected by `RandomizedSearchCV` over `uniform(0.01, 1.0)`.
Beta controls the topic-word distribution: lower values produce topics
composed of fewer, more distinctive words; higher values produce broader
topics. A value of ~0.79 produces relatively diffuse topic-word distributions,
which may be appropriate for transcribed speech where the same concept is
expressed with varied vocabulary across speakers.
*(The standard rule-of-thumb default is `beta = 0.01`.)*

**Max iterations: 5**
Set to 5 based on the `RandomizedSearchCV` search space (`[5, 10, 20]`).
The best-found configuration used 5 iterations. This is low relative to
defaults and may mean the model has not fully converged; it was retained
because the large corpus size makes additional iterations very expensive.

**Merge cutoff: 1.0 (merging disabled)**
Topics are merged if their cosine similarity exceeds this threshold.
A cutoff of 1.0 means two topics must be identical to be merged, which
effectively disables merging. The merge infrastructure is retained in the
code for future experimentation with lower thresholds.

**random_state: 42**
Fixed for reproducibility across runs.

#### Hyperparameter search (deprecated / reference only)

A `RandomizedSearchCV` block was used to find the alpha and beta values
above. The search covered:

```python
param_dist = {
    "n_components": np.arange(10, 31, 5),   # 10, 15, 20, 25, 30
    "doc_topic_prior": uniform(0.01, 1.0),  # alpha
    "topic_word_prior": uniform(0.01, 1.0), # beta
    "max_iter": [5, 10, 20],
}
# n_iter=20 random trials, cv=3-fold, scored by LDA log-likelihood
# n_jobs=-1 (all CPU cores)
```

This block is preserved but commented out in `topics.py`. The best-found
parameters are hardcoded directly into the model to avoid re-running the
search. If the corpus or preprocessing changes substantially, the search
should be re-run.

---

### Similarity

**TF-IDF vectorizer settings**
- `max_df = 0.90`: excludes terms appearing in more than 90% of meetings
- `min_df = 2`: excludes terms appearing in fewer than 2 meetings (absolute count)
- `stop_words = "english"` (sklearn built-in)
- `binary = False`: uses raw TF-IDF weights rather than binary presence/absence

**Scope: within city-state only**
Similarity is computed only between meetings in the same city-state pair,
not across all meetings. This is intentional — the measure is designed to
capture within-municipality change over time, not cross-municipality similarity.

**Election places merge**
Before embedding, the preprocessed dataset is inner-joined to an
election-places lookup file. This restricts the similarity analysis to
municipalities that are also present in the election dataset, which is the
intended unit of analysis for downstream work.

---

### Dynamic topics (BERTopic)

**Vectorizer: TF-IDF**
Same `min_df`/`max_df` settings as the LDA model (5%/95%).

**UMAP settings: `random_state=42`, `low_memory=True`**
`random_state` is fixed for reproducibility. `low_memory=True` is set
because the full meeting corpus is too large to fit comfortably in memory
during UMAP fitting.

**HDBSCAN: `min_cluster_size=150`, `prediction_data=True`**
`min_cluster_size=150` means a topic must contain at least 150 meetings
to be recognized as a cluster rather than noise. This is a relatively
high threshold chosen to ensure that identified topics represent
widespread, recurring phenomena rather than rare or idiosyncratic meetings.
`prediction_data=True` is required by BERTopic to enable probability
estimation for all documents.

**Novelty window: 2 years**
A meeting's novelty score is computed relative to the mean topic distribution
of all meetings in the same city-state in the **prior two calendar years**.
This window was chosen to capture medium-term change (long enough to
establish a baseline, short enough to be sensitive to genuine shifts).

**Topic activity threshold: 0.01**
A topic is considered "active" in a meeting if its probability exceeds 0.01.
This low threshold means a topic does not need to dominate a meeting to be
counted — it just needs to be meaningfully present.

**Stop words: same minimal list as LDA**
`["uh", "um", "okay", "ok", "yeah"]` plus NLTK English defaults.

**Minimum meeting length: 500 words**
Same filter as the LDA step.

---

## Known inconsistencies / open questions

- **LDA `num_topics` mismatch:** The configuration block in the original
  script set `num_topics = 15` at the call site, but the model was hardcoded
  to `n_components=20` based on the randomized search result. The pipeline
  currently uses 20. The `15` value should either be updated or the
  inconsistency documented with a rationale. See `topics.py`.

- **Stop word list scope:** The current minimal stop word list (`["uh", "um",
  "okay", "ok", "yeah"]`) does not suppress high-frequency procedural language
  like `council`, `motion`, `agenda`, or `vote`. Whether these terms should be
  added depends on the research question — they are corpus-noise from a
  topic-coherence perspective but may be substantively relevant for some
  analyses.

- **Sentence-level topics:** A `get_sentence_topics()` function exists in
  the code but is currently commented out at all call sites. It was designed
  to apply the fitted LDA model to the sentence-level sentiment data to
  produce a combined sentiment+topic dataset. This functionality was set aside
  but is not removed, pending a decision on whether sentence-level topic
  assignment is needed.

- **`combine` step (deprecated):** A `combine == True` branch in the original
  script merged topic and sentiment outputs and joined to the election dataset.
  This step was marked deprecated and is not included in the modular pipeline.
  Its logic should either be formalized into a `combine.py` module or
  explicitly removed.
