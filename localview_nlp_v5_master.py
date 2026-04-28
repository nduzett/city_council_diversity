# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:56:38 2024

@author: neild
"""
# general libraries

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import uniform
from scipy.special import softmax
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import pandas as pd
import numpy as np
import re
from statistics import mean
from tqdm import tqdm
import spacy
from collections import Counter
import gc
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from collections import defaultdict
import os

# preprocessing models
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from wtpsplit import SaT  # for segmenting
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')
tokenizer = SaT("sat-3l-sm")
wl = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
tqdm.pandas()

# sentiment models

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
roberta_tokenizer = AutoTokenizer.from_pretrained(model_path)
roberta_config = AutoConfig.from_pretrained(model_path)
roberta_model = AutoModelForSequenceClassification.from_pretrained(model_path)

# topic model

# textual similarity

# directories
data_directory = "C:\\Users\\Neil\\Desktop\\ccd_nlp_master\\localview_data"
save_directory = "C:\\Users\\Neil\\Desktop\\ccd_nlp_master"
embeddings_save_path = "C:\\Users\\Neil\\Desktop\\ccd_nlp_master"
last_processed_index_path = "C:\\Users\\Neil\\Desktop\\ccd_nlp_master"


# Preprocessing functions
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def remove_proper_nouns(text):
    length = len(text)
    # if text is over max length limit allowed by nlp, split it into 2 and then rejoin aftert
    if length < 1000000:
        doc = nlp(text)
        return ' '.join([token.text for token in doc if token.pos_ != 'PROPN'])
    elif length >= 1000000 and length < 2000000:
        doc1 = nlp(text[0:1000000])
        doc2 = nlp(text[1000001:])
        text1 = ' '.join(
            [token.text for token in doc1 if token.pos_ != 'PROPN'])
        text2 = ' '.join(
            [token.text for token in doc2 if token.pos_ != 'PROPN'])
        return text1 + " " + text2
    elif length > 2000000:
        return ' '

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_pauses(text):
    return re.sub("[\(\[].*?[\)\]]", "", text)

def lemmatizer(string_var):
    word_pos_tags = nltk.pos_tag(
        word_tokenize(string_var))  # Get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(
        word_pos_tags)]  # Map the position tag and lemmatize the word/token
    return " ".join(a)

def merge_election_places(meetings_to_merge):
    election_places = pd.read_csv(
        "{}\\election_places.csv".format(data_directory))

    # clean state
    state_dict = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT", "District of Columbia": "DC", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Iowa": "IA", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Massachusetts": "MA", "Maryland": "MD", "Maine": "ME", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
        "Montana": "MT", "North Carolina": "NC", "North Dakota": "ND", "Nebraska": "NE", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "Nevada": "NV", "New York": "NY", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Virginia": "VA", "Vermont": "VT", "Washington": "WA", "Wisconsin": "WI", "West Virginia": "WV", "Wyoming": "WY"}

    for key, value in state_dict.items():
        meetings_to_merge['state_name'] = meetings_to_merge['state_name'].apply(
            lambda x: x.replace(key, value))

    # clean city
    meetings_to_merge['place_name'] = meetings_to_merge['place_name'].apply(
        lambda x: x.lower())
    meetings_to_merge['place_name'] = meetings_to_merge['place_name'].apply(
        lambda x: x.strip())
    meetings_to_merge['place_name'] = meetings_to_merge['place_name'].apply(
        lambda x: x.replace(" city", ""))
    meetings_to_merge['place_name'] = meetings_to_merge['place_name'].apply(
        lambda x: x.replace(" town", ""))
    meetings_to_merge['place_name'] = meetings_to_merge['place_name'].apply(
        lambda x: x.replace(" village", ""))

    election_places['geo_name'] = election_places['geo_name'].apply(
        lambda x: x.lower())
    election_places['geo_name'] = election_places['geo_name'].apply(
        lambda x: x.strip())
    election_places['geo_name'] = election_places['geo_name'].apply(
        lambda x: x.replace(" city", ""))
    election_places['geo_name'] = election_places['geo_name'].apply(
        lambda x: x.replace(" town", ""))
    election_places['geo_name'] = election_places['geo_name'].apply(
        lambda x: x.replace(" village", ""))

    merged_df = pd.merge(election_places, meetings_to_merge, left_on=[
                         'state_abb', 'geo_name'], right_on=['state_name', 'place_name'], how='inner')

    return merged_df

def run_preprocessing():

    preprocess_time_0 = time.time()
    years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
             2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    for year in years:
        print("Appending {}...".format(year))
        meetings_year = pd.read_parquet("{}\\meetings.{}.parquet".format(
            data_directory, year), engine='pyarrow')
        meetings_year = meetings_year[meetings_year['place_govt']
                                      == 'MUNICIPAL COUNCIL']
        if year == years[0]:
            meetings_appended = meetings_year
        else:
            meetings_appended = pd.concat([meetings_appended, meetings_year])

    meetings_appended['document'] = meetings_appended['caption_text_clean']

    # doing it this early was a mistake
    # MERGE WITH ELECTION DATA HERE TO KEEP ONLY THE RELEVANT MEETINGS
    # meetings_merged = merge_election_places(meetings_appended)

    # Text preprocessing

    print("")
    print("Preprocessing...")
    print("    removing pauses")
    meetings_appended['document'] = meetings_appended['document'].progress_apply(
        remove_pauses)
    print("    converting to lowercase")
    meetings_appended['document'] = meetings_appended['document'].progress_apply(
        lambda x: x.lower())
    print("    removing punctuation")
    meetings_appended['document'] = meetings_appended['document'].progress_apply(
        remove_punctuation)
    print("    removing blank captioning")
    meetings_appended['document'] = meetings_appended['document'].progress_apply(
        lambda x: x.replace("<no caption available>", ""))
    print("    removing proper nouns")
    meetings_appended['document'] = meetings_appended['document'].progress_apply(
        remove_proper_nouns)
    print("    lemmatizing")
    meetings_appended['document'] = meetings_appended['document'].progress_apply(
        lemmatizer)

    print("")
    print("Removing blank meetings and saving...")

    meetings_appended = meetings_appended[meetings_appended['document'].str.len(
    ) > 50]
    meetings_appended['meeting_length'] = meetings_appended['document'].apply(
        lambda x: len(str(x).split()))

    print("Reduced to {} meetings.".format(len(meetings_appended)))

    meetings_appended['meeting_id'] = meetings_appended.index

    save_df = meetings_appended[['meeting_id', 'state_name',
                                 'place_name', 'meeting_date', 'document', 'meeting_length']]
    save_df.to_csv("{}\\localview_data_preprocessed.csv".format(
        save_directory), index=False, header=True)

    preprocess_time_1 = time.time()
    preprocess_time = round((preprocess_time_1 - preprocess_time_0) / 3600, 2)
    print("Preprocessing took {} hours".format(preprocess_time))

# Segmenting functions

def get_sentences(input_text):
    sentence_list = tokenizer.split(input_text)
    meeting_sentences_string = "|".join(sentence_list)
    return meeting_sentences_string

def run_segmenter(input_df, chunk_size, offset_chunks):  # 17,046 meetings to segment

    for start in range(chunk_size * offset_chunks, len(input_df), chunk_size):
        end = start + chunk_size
        chunk = input_df.iloc[start:end]

        segmenting_time_0 = time.time()
        print("")
        print("Segmenting chunk {}...".format(start / chunk_size))
        chunk['sentences'] = chunk['document'].progress_apply(
            lambda x: get_sentences(x))

        # create meeting ids
        segmenting_time_1 = time.time()

        segmenting_time = round(
            (segmenting_time_1 - segmenting_time_0) / 3600, 2)
        print("Segmenting chunk {} took {} hours".format(
            start / chunk_size, segmenting_time))

        if start == 0:
            chunk.to_csv("{}\\localview_data_segmented.csv".format(
                save_directory), index=False, header=True)
        else:
            chunk.to_csv("{}\\localview_data_segmented.csv".format(
                save_directory), index=False, header=False, mode="a")

# Sentiment functions

def vader_sentiment_scores(text):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(text)
    return [sentiment_dict['neg'] * 100, sentiment_dict['neu'] * 100, sentiment_dict['pos'] * 100]

def textblob_sentiment_scores(text):
    polarity = TextBlob(text).sentiment.polarity
    subjectivity = TextBlob(text).sentiment.subjectivity
    return [polarity, subjectivity]

def roberta_sentiment_scores(text):

    try:
        encoded_input = roberta_tokenizer(text, return_tensors='pt')
        roberta_output = roberta_model(**encoded_input)
        roberta_scores = roberta_output[0][0].detach().numpy()
        roberta_scores = softmax(roberta_scores)  # 0, 1, 2 neg, neu, pos
        return [roberta_scores[0] * 100, roberta_scores[1] * 100, roberta_scores[2] * 100]
    except:
        return [np.nan, np.nan, np.nan]

def get_sentiment(df, offset):

    num_meetings = len(df['sentences'])
    times = []
    j = 0 + offset

    # Run Sentiment Analysis
    while j < num_meetings:
        time_0 = time.time()
        if j != offset:
            time_remaining = round(mean(times) * (num_meetings - j) / 3600, 2)
            print("{} % complete, {} meetings done, {} hours remaining".format(
                round(100 * j / num_meetings, 2), j, time_remaining))

        meeting_sentences = df['sentences'][j].split("|")
        meeting_chunk = []
        for sentence in meeting_sentences:
            sentence_scores = []

            # get vader scores
            sentence_scores.extend(vader_sentiment_scores(sentence))

            # get textblob scores
            sentence_scores.extend(textblob_sentiment_scores(sentence))

            # get roberta scores
            sentence_scores.extend(roberta_sentiment_scores(sentence))

            meeting_chunk.append([df['meeting_id'][j], df['state_name'][j], df['place_name']
                                 [j], df['meeting_date'][j], sentence, "|".join(map(str, sentence_scores))])

        try:

            # write to csv as we go to save progress in case of crash (file takes many hours to run)
            save_df = pd.DataFrame(meeting_chunk, columns=[
                                   'meeting_id', 'state', 'city', 'meeting_date', 'sentence', 'scores'])

            if j == 0:
                save_df.to_csv("{}\\sentence_sentiment_scores.csv".format(
                    save_directory), index=False)
            else:
                save_df.to_csv("{}\\sentence_sentiment_scores.csv".format(
                    save_directory), index=False, header=False, mode='a')

        except Exception as e:
            print(e)
            continue
        time_1 = time.time()
        times.append(time_1 - time_0)
        j += 1

# Topic functions

def merge_topics(lda_model, num_topics, threshold=0.8):

    # Get the topic-word distributions (components) from the LDA model
    topic_word_distributions = lda_model.components_ / \
        lda_model.components_.sum(axis=1)[:, np.newaxis]

    # Compute cosine similarity between all topics
    similarity_matrix = cosine_similarity(topic_word_distributions)

    # Mark topics to be merged if their cosine similarity is above the threshold
    merged_groups = []
    visited = set()

    for i in range(num_topics):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i+1, num_topics):
            if j in visited:
                continue
            if similarity_matrix[i, j] >= threshold:
                group.append(j)
                visited.add(j)
        merged_groups.append(group)

    return merged_groups

def merge_topic_distributions(merged_groups, doc_topic_matrix):

    new_doc_topic_matrix = np.zeros(
        (doc_topic_matrix.shape[0], len(merged_groups)))

    for new_topic_idx, group in enumerate(merged_groups):
        # Sum the probabilities of the original topics that have been merged
        new_doc_topic_matrix[:, new_topic_idx] = np.sum(
            doc_topic_matrix[:, group], axis=1)

    return new_doc_topic_matrix

def print_merged_topics(lda_model, vectorizer, merged_groups, top_n_words):
    f = open("{}\\merged_topics.txt".format(save_directory), "w")
    # Get the feature names (vocabulary) from the vectorizer
    words = vectorizer.get_feature_names_out()

    # Get the topic-word distributions (components) from the LDA model
    topic_word_distributions = lda_model.components_ / \
        lda_model.components_.sum(axis=1)[:, np.newaxis]

    print("Merged Topics:")
    f.write("Merged Topics:")

    for new_topic_idx, group in enumerate(merged_groups):
        # Merge the topic-word distributions by summing them
        merged_topic_dist = np.sum(topic_word_distributions[group], axis=0)

        # Get the indices of the top words for the merged topic
        top_word_indices = merged_topic_dist.argsort()[-top_n_words:][::-1]
        top_words = [words[i] for i in top_word_indices]

        # Print the merged topic and its top words
        print(f"Merged Topic {new_topic_idx}: {', '.join(top_words)}")
        print("")
        f.write(f"Merged Topic {new_topic_idx}: {', '.join(top_words)}")
        f.write("")
    f.close()

def compute_topic_proportions(tokens, topics):  # DEPRECATED
    print(tokens)
    word_counts = Counter(tokens)
    topic_proportions = {}
    total_words = len(tokens)

    for topic in topics:
        topic_word_count = sum(word_counts[word] for word in topics[topic])
        topic_proportions[topic] = topic_word_count / \
            total_words if total_words > 0 else 0
        print(topic_proportions)

    return topic_proportions

# Custom scorer for LDA log-likelihood
def log_likelihood_scorer(estimator, X):
    """Compute log-likelihood (higher is better)."""
    return estimator.score(X)

def get_topics(df, min_doc_freq, max_doc_freq, num_topics, alpha, beta, num_top_words, merge_cutoff, more_stop_words):

    """
    split data variable, keep only pre 2021
    """
    df['meeting_date'] = df['meeting_date'].apply(lambda x: str(x))
    df['meeting_year'] = df['meeting_date'].apply(lambda x: int(x.split("-")[0]))

    """
    print(df['meeting_year'].unique())
    print(len(df))
    df = df[df['meeting_year'] < 2021]
    print(df['meeting_year'].unique())
    print(len(df))
    print(df.head())
    """

    documents = df['document']

    print("")
    print("Vectorizing...")

    # Vectorization: min_df and max_df remove keywords that appear too frequently or too rarely
    # vectorizer = TfidfVectorizer(stop_words = stop_words, min_df = min_doc_freq, max_df = max_doc_freq)
   # Define stop words
    stop_words = list(stopwords.words('english')) + more_stop_words
    
    # Vectorization
    vectorizer_time_0 = time.time()
    
    vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=min_doc_freq, max_df=max_doc_freq)
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    vectorizer_time_1 = time.time()
    vectorizer_time = round((vectorizer_time_1 - vectorizer_time_0) / 3600, 2)
    print("Vectorizing took {} hours".format(vectorizer_time))
    """
    print("\nPerforming Randomized Search for best hyperparameters...")
    search_time_0 = time.time()
    
    # Define hyperparameter search space
    param_dist = {
        "n_components": np.arange(10, 31, 5),  # Number of topics (10 to 50)
        "doc_topic_prior": uniform(0.01, 1.0),  # Alpha (document-topic prior)
        "topic_word_prior": uniform(0.01, 1.0),  # Beta (word-topic prior)
        "max_iter": [5, 10, 20],  # Number of iterations
    }
    
    # Initialize base LDA model
    lda_model = LatentDirichletAllocation(random_state=42)
    
    # Perform Randomized Search
    random_search = RandomizedSearchCV(
        lda_model,
        param_distributions=param_dist,
        n_iter=20,  # Number of random trials
        scoring=make_scorer(log_likelihood_scorer),  # Optimize log-likelihood
        n_jobs=-1,  # Use all CPUs
        cv=3,  # 3-fold cross-validation
        random_state=42
    )
    
    # Fit Randomized Search on document-term matrix
    random_search.fit(doc_term_matrix)
    
    search_time_1 = time.time()
    search_time = round((search_time_1 - search_time_0) / 3600, 2)
    print("Finding best hyperparameters took {} hours".format(search_time))
    
    # Get best hyperparameters
    best_params = random_search.best_params_
    print("\nBest Parameters Found:", best_params)
    
    # Use the best parameters to fit the final LDA model
    print("\nFitting the best LDA model...")
    
    model_time_0 = time.time()
    
    lda_model = LatentDirichletAllocation(
        n_components=best_params["n_components"],
        doc_topic_prior=best_params["doc_topic_prior"],
        topic_word_prior=best_params["topic_word_prior"],
        max_iter=best_params["max_iter"],
        random_state=42
    )
    """
    model_time_0 = time.time()
    
    lda_model = LatentDirichletAllocation(
        n_components=20,
        doc_topic_prior=0.3845401188473625,
        topic_word_prior=0.7896910002727693,
        max_iter=5,
        random_state=42
    )
    
    lda_model.fit(doc_term_matrix)
    
    model_time_1 = time.time()
    model_time = round((model_time_1 - model_time_0) / 3600, 2)
    print("Fitting the LDA model took {} hours".format(model_time))
    
    # Proceed with merging topics and saving results
    print("\nMerging similar topics...")
    
    merged_groups = merge_topics(lda_model, 20, threshold=merge_cutoff)
    doc_topic_matrix = lda_model.transform(doc_term_matrix)  # Document-topic distributions
    merged_doc_topic_matrix = merge_topic_distributions(merged_groups, doc_topic_matrix)
    
    print_merged_topics(lda_model, vectorizer, merged_groups, num_top_words)
    
    print("\nSaving merged topics and document probabilities...")
    
    df_merged_topics = pd.DataFrame(
        merged_doc_topic_matrix, 
        columns=[f'merged_topic_{i}' for i in range(len(merged_groups))]
    )
    
    df['merge_string'] = df['document'].str[:50]
    save_df = df[['state_name', 'place_name', 'meeting_date', 'meeting_length', 'merge_string']].copy()
    
    for x in range(len(merged_groups)):
        save_df[f'topic_{x}_probs'] = df_merged_topics[f'merged_topic_{x}']
    
    save_df.to_csv(f"{save_directory}/topic_data_final.csv", index=False, header=True)
    
    print("\nDone saving.\nPrinting topic means...\n")
    
    for x in range(len(merged_groups)):
        print(f"Topic {x}: {df_merged_topics[f'merged_topic_{x}'].mean()}")

    # Delete old dataframes to free up memory
    dfs_to_delete = [df, save_df, df_merged_topics]
    del df
    del save_df
    del df_merged_topics
    del dfs_to_delete

    #get_sentence_topics(sentiment_df, vectorizer, lda_model)

    """ DEPRECATED
    # Get dictionary of topics
    feature_names = vectorizer.get_feature_names_out()
    #topic_word_matrix = lda_model.components_

    topic_dict = {}

    for topic_idx, topic_weights in enumerate(merged_groups):
        top_words_indices = topic_weights.argsort()[::-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topic_dict[f"topic_{topic_idx}"] = top_words

    with open("topics.json", "w") as file:
        json.dump(topic_dict, file)

    print(len(topic_dict))
    """

def get_sentence_topics(sentiment_df, vectorizer, lda_model):
    """
    SENTENCE LEVEL
    """
    print("")
    print("Calculating sentence-level topic distributions...")
    print("")

    # Transform sentences in the DataFrame

    print("Vectorizing...")
    print("")
    sentence_term_matrix = vectorizer.transform(sentiment_df["sentence"])

    print("Merging topics...")
    print("")
    merged_groups = merge_topics(lda_model, num_topics, threshold=merge_cutoff)

    print("Transforming sentence matrix...")
    print("")
    sentence_topic_matrix = lda_model.transform(
        sentence_term_matrix)  # Document-topic distributions
    merged_sentence_topic_matrix = merge_topic_distributions(
        merged_groups, sentence_topic_matrix)

    # Add topic distributions as new columns
    print("Saving...")
    print("")

    topic_columns = [f"topic_{i}" for i in range(len(merged_groups))]
    sentiment_df[topic_columns] = pd.DataFrame(
        merged_sentence_topic_matrix, index=sentiment_df.index)

    sentiment_df.to_csv("{}\\sentiment_and_topic_sentences.csv".format(
        save_directory), index=False, header=True)

# similarity functions

def count_unique_words(text):
    # Tokenize the text (split by whitespace)
    words = text.split()

    # Return the count of unique words
    return len(set(words))  # Using a set to count unique words

def vocab_embed(documents_df):
    # AXE THIS; JUST DO A BINARY 0-1 VOCAB VECTOR FOR ALL WORDS ACROSS MEETINGS

    documents = documents_df['document']
    print("Calculating high-frequency words to remove...")

    # Identify words appearing in >80% of documents or < 1% of documents
    vectorizer = TfidfVectorizer(max_df = 0.90, min_df = 2, stop_words="english", binary = False)
    tfidf_matrix = vectorizer.fit_transform(documents)
                                            
    tfidf_vectors = ["|".join(map(str, row.toarray().flatten())) for row in tfidf_matrix]

    # Add new column to the DataFrame
    documents_df = documents_df.copy()  # Avoid modifying original DataFrame
    documents_df["tfidf_vector"] = tfidf_vectors

    print("TF-IDF vectors added to dataframe")

    return documents_df

def compute_similarity(index, df, starting_row):
    """Computes similarity of a meeting with past meetings in the same city/state."""
    current_date = df.loc[index, "meeting_date_clean"]
    current_city = df.loc[index, "place_name"]
    current_state = df.loc[index, "state_name"]

    # Filter past meetings in the same city and state
    past_meetings = df[
        (df["state_name"] == current_state) & 
        (df["place_name"] == current_city) & 
        (df["meeting_date_clean"] < current_date)
    ].copy()

    if past_meetings.empty:
        similarities = []
        past_dates = []
    else:
        # Convert current meeting embedding from string to NumPy array
        current_embedding = np.array([float(x) for x in df.loc[index, "tfidf_vector"].split('|')]).reshape(1, -1)

        # Convert past meetings' embeddings
        past_meetings["np_array"] = past_meetings["tfidf_vector"].apply(lambda x: np.array([float(i) for i in x.split('|')]))

        # Compute cosine similarity
        past_meetings["similarity"] = past_meetings["np_array"].apply(lambda x: cosine_similarity(current_embedding, x.reshape(1, -1))[0][0])

        # Format similarity scores and dates
        similarities = past_meetings["similarity"].apply(lambda x: f"{x:.4f}").tolist()
        past_dates = past_meetings["meeting_date_clean"].astype(str).tolist()

    # Save results as a DataFrame
    save_df = pd.DataFrame({
        "state_name": [current_state],
        "place_name": [current_city],
        "meeting_date_clean": [current_date],
        "meeting_length": [df.loc[index, "meeting_length"]],
        "unique_word_count": [df.loc[index, "unique_word_count"]],
        "similarities": ["|".join(similarities)],
        "past_meeting_dates": ["|".join(past_dates)]
    })

    # Write to file, handling headers correctly
    if index + starting_row == 0:
        save_df.to_csv(output_file, mode="w", index=False, header=True)
    else:
        save_df.to_csv(output_file, mode="a", index=False, header=False)

# Combining functions

def classify_neg_vd(row):
    if row['vader_neg'] > 33 and row['vader_neg'] > row['vader_neu'] and row['vader_neg'] > row['vader_pos']:
        return 1
    else:
        return 0

def classify_neu_vd(row):
    if row['vader_neu'] > 33 and row['vader_neu'] > row['vader_neg'] and row['vader_neu'] > row['vader_pos']:
        return 1
    else:
        return 0

def classify_pos_vd(row):
    if row['vader_pos'] > 33 and row['vader_pos'] > row['vader_neu'] and row['vader_pos'] > row['vader_neg']:
        return 1
    else:
        return 0

def classify_neg_tb(row):
    if row['tb_polarity'] <= -33:
        return 1
    else:
        return 0

def classify_neu_tb(row):
    if row['tb_polarity'] > -33 and row['tb_polarity'] < 33:
        return 1
    else:
        return 0

def classify_pos_tb(row):
    if row['tb_polarity'] >= 33:
        return 1
    else:
        return 0

def classify_neg_rob(row):
    if row['rob_neg'] > 33 and row['rob_neg'] > row['rob_neu'] and row['rob_neg'] > row['rob_pos']:
        return 1
    else:
        return 0

def classify_neu_rob(row):
    if row['rob_neu'] > 33 and row['rob_neu'] > row['rob_neg'] and row['rob_neu'] > row['rob_pos']:
        return 1
    else:
        return 0

def classify_pos_rob(row):
    if row['rob_pos'] > 33 and row['rob_pos'] > row['rob_neu'] and row['rob_pos'] > row['rob_neg']:
        return 1
    else:
        return 0

def make_merge_string(row):
    return row['sentence'][:50]


"""
MAIN CONFIGURATION
"""
preprocess = False
segment = False
sentiment = False
topics = False
clean_sentiment = False
extract_sentiment_sentences = False
combine = False  # DEPRECATED
embedding = False
similarity = False
clean_similarity = False
dynamic_topics = True

if preprocess == True:
    run_preprocessing()

if segment == True:
    df = pd.read_csv("{}\\localview_data_preprocessed.csv".format(
        save_directory), header=0)
    offset_chunks = 0  # chunk to start at
    chunk_size = 1000
    run_segmenter(df, chunk_size, offset_chunks)

if sentiment == True:
    master_df = pd.read_csv(
        "{}\\localview_data_segmented.csv".format(save_directory), header=0)
    offset = 0
    get_sentiment(master_df, offset)

if topics == True:

    # full_df = pd.read_csv("{}\\localview_data_preprocessed.csv".format(save_directory), header = 0)
    full_df = pd.read_csv("{}\\localview_data_preprocessed.csv".format(save_directory), header=0)
    #sentiment_df = pd.read_csv("{}\\sentence_sentiment_scores.csv".format(save_directory), header=0)
    
    full_df['meeting_length'] = full_df['document'].apply(lambda x: len(str(x).split()))
    full_df = full_df[full_df['meeting_length'] >= 500]

    # Customization for topic analysis
    min_doc_freq = 0.05
    max_doc_freq = 0.95
    num_topics = 15
    alpha = 2 / num_topics  # low alpha -> documents contain fewer topics, high alpha -> documents contain more topics, good start is 1 / num_topics, bigger for longer documents
    beta = 0.01  # low beta -> topics are made up of few words in the document, high beta -> topics are made up of many words in the document
    num_top_words = 20
    merge_cutoff = 1
    #more_stop_words = ["nt", "uh", "um", "000", "know", "get", "would", "city", "think", "like", "say", "thank", "want", "yes", "well", "come", "year", "make", "right", "sioux", "dakota", "ah", "oh", "good", "ok", "eight"]
    #more_stop_words = ["nt", "uh", "um", "000", "know", "get", "would", "city", "think", "like", "say", "thank", "want", "yes", "well", "come", "year", "make", "right", "sioux", "dakota", "ah", "oh", "good", "ok", "eight", "hello", "today", "thing", "okay", "time", "look", "work", "see", "people", "need", "council", "question", "take", "really", "yeah", "two", "also", "could", "back", "talk", "na", "mean", "gon"]
    more_stop_words = ["uh", "um", "okay", "ok", "yeah"]
    
    #get_topics(full_df, sentiment_df,  min_doc_freq, max_doc_freq, num_topics, alpha, beta, num_top_words, merge_cutoff, more_stop_words)
    get_topics(full_df, min_doc_freq, max_doc_freq, num_topics, alpha, beta, num_top_words, merge_cutoff, more_stop_words)

if clean_sentiment == True:
    sentiment_df = pd.read_csv(
        "{}\\sentence_sentiment_scores.csv".format(save_directory), header=0)

    # split scores column
    print("Splitting scores column...")
    score_columns = sentiment_df['scores'].str.split('|', expand=True)
    score_columns.rename(columns={0: 'vader_neg', 1: 'vader_neu', 2: 'vader_pos', 3: 'tb_polarity', 4: 'tb_subjectivity', 5: 'rob_neg', 6: 'rob_neu', 7: 'rob_pos'}, inplace=True)
    sentiment_df = pd.concat([sentiment_df, score_columns], axis=1)

    # recast string scores as integers
    print("Converting scores stored as strings into floats...")
    sentiment_df['vader_neg'] = pd.to_numeric(
        sentiment_df['vader_neg'], errors='coerce').fillna(0).astype(np.float64)
    sentiment_df['vader_neu'] = pd.to_numeric(
        sentiment_df['vader_neu'], errors='coerce').fillna(0).astype(np.float64)
    sentiment_df['vader_pos'] = pd.to_numeric(
        sentiment_df['vader_pos'], errors='coerce').fillna(0).astype(np.float64)
    sentiment_df['tb_polarity'] = pd.to_numeric(
        sentiment_df['tb_polarity'], errors='coerce').fillna(0).astype(np.float64)
    sentiment_df['tb_subjectivity'] = pd.to_numeric(
        sentiment_df['tb_subjectivity'], errors='coerce').fillna(0).astype(np.float64)
    sentiment_df['rob_neg'] = pd.to_numeric(
        sentiment_df['rob_neg'], errors='coerce').fillna(0).astype(np.float64)
    sentiment_df['rob_neu'] = pd.to_numeric(
        sentiment_df['rob_neu'], errors='coerce').fillna(0).astype(np.float64)
    sentiment_df['rob_pos'] = pd.to_numeric(
        sentiment_df['rob_pos'], errors='coerce').fillna(0).astype(np.float64)

    # classifying sentences with 34% or more while being the highest as that sentiment
    print("Classifying sentences based on sentiment scores...")
    print("    vader...")
    sentiment_df['vd_negative'] = sentiment_df.progress_apply(
        classify_neg_vd, axis=1)
    sentiment_df['vd_neutral'] = sentiment_df.progress_apply(
        classify_neu_vd, axis=1)
    sentiment_df['vd_positive'] = sentiment_df.progress_apply(
        classify_pos_vd, axis=1)
    print("    textblob...")
    sentiment_df['tb_polarity'] = 100 * sentiment_df['tb_polarity']
    sentiment_df['tb_negative'] = sentiment_df.progress_apply(
        classify_neg_tb, axis=1)
    sentiment_df['tb_neutral'] = sentiment_df.progress_apply(
        classify_neu_tb, axis=1)
    sentiment_df['tb_positive'] = sentiment_df.progress_apply(
        classify_pos_tb, axis=1)
    print("    roberta...")
    sentiment_df['rob_negative'] = sentiment_df.progress_apply(
        classify_neg_rob, axis=1)
    sentiment_df['rob_neutral'] = sentiment_df.progress_apply(
        classify_neu_rob, axis=1)
    sentiment_df['rob_positive'] = sentiment_df.progress_apply(
        classify_pos_rob, axis=1)

    sentiment_df.to_csv("{}\\sentence_sentiment_scores_clean.csv".format(save_directory), index=False, header=True)

    # collapse sentiment DF to meeting level
    print("Collapsing to meeting level...")
    meeting_sentiment = sentiment_df[['meeting_id', 'state', 'city', 'meeting_date', 'sentence', 'vd_negative', 'vd_neutral',
                                      'vd_positive', 'tb_negative', 'tb_neutral', 'tb_positive', 'rob_negative', 'rob_neutral', 'rob_positive', 'tb_subjectivity']]

    collapsed_df = meeting_sentiment.groupby('meeting_id', as_index=False).agg({
        'state': 'first', 'city': 'first', 'meeting_date': 'first', 'sentence': 'first',
        'vd_negative': 'mean', 'vd_neutral': 'mean', 'vd_positive': 'mean',
        'tb_negative': 'mean', 'tb_neutral': 'mean', 'tb_positive': 'mean',
        'rob_negative': 'mean', 'rob_neutral': 'mean', 'rob_positive': 'mean',
        'tb_subjectivity': 'mean'
    })

    # keep first sentence of each meeting, cut to 20 characters
    collapsed_df['merge_string'] = collapsed_df.progress_apply(
        make_merge_string, axis=1)

    print("Saving...")
    collapsed_df.to_csv("{}\\sentiment_meeting_level.csv".format(save_directory), index=False, header=True)

if extract_sentiment_sentences == True:
    sentiment_df = pd.read_csv("{}\\sentence_sentiment_scores_clean.csv".format(save_directory), header=0)
    sentiment_df_small = sentiment_df.sample(n = 5000, random_state = 42)
    sentiment_df_small.to_csv("{}\\example_sentiment_scores.csv".format(save_directory), index=False, header=True)

if combine == True:  # DEPRECATED
    print("Prepping topic data...")
    df1 = pd.read_csv("{}\\topic_data_final.csv".format(
        save_directory), header=0)
    df1 = merge_election_places(df1)
    collapsed_df1 = df1.groupby('meeting_date', as_index=False).agg({
        'state_name': 'first', 'place_name': 'first', 'meeting_length': 'sum', 'merge_string': 'first',
        'topic_0_probs': 'mean', 'topic_1_probs': 'mean', 'topic_2_probs': 'mean', 'topic_3_probs': 'mean', 'topic_4_probs': 'mean', 'topic_5_probs': 'mean',
        'topic_6_probs': 'mean', 'topic_7_probs': 'mean', 'topic_8_probs': 'mean', 'topic_9_probs': 'mean', 'topic_10_probs': 'mean', 'topic_11_probs': 'mean',
        'topic_12_probs': 'mean', 'topic_13_probs': 'mean', 'topic_14_probs': 'mean', 'topic_15_probs': 'mean'
    })

    print("Prepping sentiment data...")
    df2 = pd.read_csv("{}\\sentiment_meeting_level.csv".format(
        save_directory), header=0)
    df2['state_name'] = df2['state']
    df2['place_name'] = df2['city']
    collapsed_df2 = df2.groupby('meeting_date', as_index=False).agg({
        'state_name': 'first', 'place_name': 'first', 'meeting_id': 'first', 'sentence': 'first',
        'vd_negative': 'mean', 'vd_neutral': 'mean', 'vd_positive': 'mean',
        'tb_negative': 'mean', 'tb_neutral': 'mean', 'tb_positive': 'mean',
        'rob_negative': 'mean', 'rob_neutral': 'mean', 'rob_positive': 'mean',
        'tb_subjectivity': 'mean'
    })
    df2 = merge_election_places(df2)

    print("Merging...")
    # Step 1: Initial merge on 'state_name', 'place_name', and 'meeting_date'
    initial_merge = pd.merge(
        collapsed_df1, collapsed_df2,
        on=['state_name', 'place_name', 'meeting_date'],
        how='inner',
        suffixes=('_df1', '_df2'),
        indicator=True
    )
    initial_merge.to_csv("{}\\initial_merge.csv".format(save_directory), index=False, header=True)

if embedding == True:
    df = pd.read_csv("{}\\localview_data_preprocessed.csv".format(
        save_directory), header = 0)
    #df = df.iloc[:500]

    # only embed meetings that will eventually merge to election dataset
    merged_df = merge_election_places(df)
    del df
    
    print("Calculating meeting length...")
    merged_df['meeting_length'] = merged_df['document'].progress_apply(
        lambda x: len(str(x).split()))
    print("Calculating unique word counts...")
    merged_df['unique_word_count'] = merged_df['document'].progress_apply(count_unique_words)
    
    print("Embedding...")
    
    save_df = vocab_embed(merged_df)
    save_df.to_csv("{}\\meeting_embeddings.csv".format(save_directory), index = False, header = True)
    print("Saved embeddings dataframe")
    
    del merged_df
    del save_df
    gc.collect()

if similarity == True:
    input_file = f"{save_directory}\\meeting_embeddings.csv"
    output_file = f"{save_directory}\\meeting_similarity_data.csv"
    
    # Define the starting row index
    starting_row = 0

    # Load full dataset (only relevant columns)
    full_df = pd.read_csv(input_file, usecols=["state_name", "place_name", "meeting_date", "meeting_length", "unique_word_count", "tfidf_vector"], skiprows = range(1, starting_row + 1))
    
    # 🔹 SORT BY STATE, CITY, AND DATE
    full_df["meeting_date_clean"] = pd.to_datetime(full_df["meeting_date"])
    full_df = full_df.sort_values(by=["state_name", "place_name", "meeting_date_clean"]).reset_index(drop = True)

    full_df.progress_apply(lambda row: compute_similarity(row.name, full_df, starting_row), axis = 1)

if clean_similarity == True:
    df = pd.read_csv(f"{save_directory}\\meeting_similarity_data.csv")
    print("Splitting...")
    df["sim_score"] = df["similarities"].str.split("|")
    df["sim_date"] = df["past_meeting_dates"].str.split("|")
    print("Exploding...")
    df = df.explode(["sim_score", "sim_date"], ignore_index = True)
    
    save_df = pd.DataFrame({
        "state_name": df['state_name'],
        "place_name": df['place_name'],
        "meeting_date_clean": df['meeting_date_clean'],
        "meeting_length": df["meeting_length"],
        "unique_word_count": df["unique_word_count"],
        "sim_score": df['sim_score'],
        "sim_date": df['sim_date']
    })
    
    save_df.to_csv(f"{save_directory}\\meeting_similarity_data_split.csv", mode = "w", index = False, header = True)
    
if dynamic_topics == True:
    print("Loading data...")
    save_directory = "C:\\Users\\Neil\\Desktop\\ccd_nlp_master"
    min_doc_freq = 0.05
    max_doc_freq = 0.95
    
    df = pd.read_csv(f"{save_directory}\\localview_data_preprocessed.csv", header=0)
    
    print("Cleaning data...")
    df['meeting_length'] = df['document'].apply(lambda x: len(str(x).split()))
    df = df[df['meeting_length'] >= 500]
    df['meeting_date'] = df['meeting_date'].astype(str)
    df['meeting_year'] = df['meeting_date'].apply(lambda x: int(x.split("-")[0]))
    
    more_stop_words = ["uh", "um", "okay", "ok", "yeah"]
    stop_words = list(stopwords.words('english')) + more_stop_words
    
    # Keep relevant columns including state_name
    df = df[["document", "meeting_date", "meeting_year", "place_name", "state_name"]]
    
    # Create combined city-state key
    df["city_state"] = df["place_name"] + ", " + df["state_name"]
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=min_doc_freq, max_df=max_doc_freq)
    embeddings = vectorizer.fit_transform(df['document'])
    
    print("Fitting model...")
    umap_model = UMAP(random_state=42, low_memory=True)
    hdbscan_model = HDBSCAN(min_cluster_size=150, prediction_data=True)
    topic_model = BERTopic(calculate_probabilities=True, hdbscan_model=hdbscan_model, umap_model=umap_model, low_memory=True)
    topics, probs = topic_model.fit_transform(df['document'], embeddings)
    
    # Save the topics
    output_path = os.path.join(save_directory, "topic_keywords_top10.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for topic_id in topic_model.get_topics():
            words_scores = topic_model.get_topic(topic_id)
            if words_scores is None:
                continue
            f.write(f"Topic {topic_id}:\n")
            for word, score in words_scores[:10]:
                f.write(f"  {word}: {score:.4f}\n")
            f.write("\n")
    print(f"Top 10 topic keywords saved to: {output_path}")
    
    # Topic probabilities matrix
    topic_probs = np.array(probs, dtype=object)
    topic_probs = np.stack(topic_probs)
    num_topics = topic_probs.shape[1]
    
    # Build doc_info with city_state
    doc_info = pd.DataFrame({
        "meeting_date": df["meeting_date"].values,
        "meeting_year": df["meeting_year"].values,
        "place_name": df["place_name"].values,
        "state_name": df["state_name"].values,
        "city_state": df["city_state"].values
    })
    
    # Add topic shares
    topic_share_cols = [f"topic_{i}_share" for i in range(num_topics)]
    topic_share_df = pd.DataFrame(topic_probs, columns=topic_share_cols)
    doc_info = pd.concat([doc_info, topic_share_df], axis=1)
    
    # === Calculate Novelty Scores Within City-State ===
    print("Calculating novelty scores within each city-state...")
    novelty_scores = []
    similarities = []
    
    for i in range(len(doc_info)):
        if i % 100 == 0:
            print(f"{i}/{len(doc_info)}")
        current_time = doc_info.loc[i, "meeting_year"]
        current_city_state = doc_info.loc[i, "city_state"]
        current_vector = topic_probs[i].reshape(1, -1)
    
        window_start = current_time - 2
        in_window = (
            (doc_info["city_state"] == current_city_state) &
            (doc_info["meeting_year"] >= window_start) &
            (doc_info["meeting_year"] < current_time)
        )
        earlier_vectors = topic_probs[in_window]
    
        if earlier_vectors.shape[0] == 0:
            similarities.append(None)
            novelty_scores.append(None)
            continue
    
        mean_earlier_vector = earlier_vectors.mean(axis=0).reshape(1, -1)
        sim = cosine_similarity(current_vector, mean_earlier_vector)[0][0]
        novelty = 1 - sim
    
        similarities.append(sim)
        novelty_scores.append(novelty)
    
    doc_info["novelty_score_2yr_city"] = novelty_scores
    
    # === Count New Topics Within City-State ===
    print("Counting new topics within each city-state...")
    doc_info["topic"] = topics
    threshold = 0.01
    
    # Step 1: Collect topic sets by city-state and year
    city_year_topicsets = {}
    for (city_state, year), group in doc_info.groupby(["city_state", "meeting_year"]):
        indices = group.index
        all_topics = set()
        for i in indices:
            probs = topic_probs[i]
            active = set(np.where(probs > threshold)[0]) - {-1}
            all_topics.update(active)
        city_year_topicsets[(city_state, year)] = all_topics
    
    # Step 2: Identify new topics
    new_topics_by_city_year = {}
    for (city_state, year) in city_year_topicsets.keys():
        prior_topics = set()
        for y in range(year - 2, year):
            prior_topics.update(city_year_topicsets.get((city_state, y), set()))
        current_topics = city_year_topicsets.get((city_state, year), set())
        new_topics = current_topics - prior_topics
        new_topics_by_city_year[(city_state, year)] = new_topics
    
    # Step 3: Count new topics per meeting
    new_topic_counts = []
    for i, row in doc_info.iterrows():
        if i % 100 == 0:
            print(f"Processing meeting {i+1:,} of {len(doc_info):,}")
    
        city_state = row["city_state"]
        year = row["meeting_year"]
    
        if pd.isna(row["novelty_score_2yr_city"]):
            new_topic_counts.append(None)
            continue
    
        probs = topic_probs[i]
        active_topics = set(np.where(probs > threshold)[0]) - {-1}
        new_topics = new_topics_by_city_year.get((city_state, year), set())
        count_new = len(active_topics & new_topics)
        new_topic_counts.append(count_new)
    
    doc_info["num_new_topics_city"] = new_topic_counts
    
    # === Save output with separate city and state columns ===
    output_path = os.path.join(save_directory, "meeting_topic_novelty_by_city.csv")
    doc_info.to_csv(output_path, index=False, columns=[
        "meeting_date", "meeting_year", "place_name", "state_name", "city_state",
        "novelty_score_2yr_city", "num_new_topics_city"
    ] + topic_share_cols)
    print(f"Saved results to: {output_path}")

























