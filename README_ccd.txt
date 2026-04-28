# LocalView NLP & City Council Network Scripts

## Files

### `localview_nlp_v5_master.py`
A pipeline for NLP analysis of local government meeting transcripts from the [LocalView dataset](https://doi.org/10.7910/DVN/TQBZLZ). The script is structured as a series of toggleable modules that can be run independently:

- **Preprocessing** — Loads municipal council meeting transcripts (2006–2023), cleans text (removes pauses, punctuation, proper nouns), lemmatizes, and saves a preprocessed CSV.
- **Segmentation** — Splits meeting documents into sentences using the `wtpsplit` SaT model.
- **Sentiment Analysis** — Scores each sentence with three models: VADER, TextBlob, and a RoBERTa-based transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
- **Topic Modeling** — Supports both LDA (via scikit-learn TF-IDF + `LatentDirichletAllocation`) and BERTopic (via UMAP + HDBSCAN). Includes topic merging utilities and hyperparameter search scaffolding (currently commented out).
- **Textual Similarity** — Computes cosine similarity between meetings within the same city, enabling detection of repeated or boilerplate agenda content.
- **Dynamic/Novelty Topics** — Uses BERTopic probabilities to compute a novelty score (how much a meeting's topic distribution diverges from recent prior meetings in the same city) and counts newly emerging topics per city per year.

**Key dependencies:** `transformers`, `bertopic`, `hdbscan`, `umap-learn`, `wtpsplit`, `spacy`, `textblob`, `vaderSentiment`, `scikit-learn`, `nltk`, `pandas`, `numpy`

---

### `make_city_council_historical_network_v6.py`
Constructs a longitudinal record of who sits on local governing bodies (city councils, county legislatures, or school boards) in each year, using electoral results data. Designed to work with the LEDB (Local Election Database) candidate-level CSVs.

- Filters to governing bodies that have at least some at-large elections.
- For each city, iterates year by year, maintaining a rolling "sitting council" dictionary that maps districts to current officeholders (by candidate ID).
- Seats not updated within 4 election cycles are dropped, approximating term expiration.
- Detects the first year in which the council's seat count stabilizes across three consecutive election years ("year full"), to flag the burn-in period before the panel data is reliable.
- Outputs two CSVs: a flat sitting-council file and a year-full file per city.

**Key dependencies:** `pandas`, `copy`
