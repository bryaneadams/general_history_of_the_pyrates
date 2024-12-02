import string, re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer


def pre_process_pyrates(text: str, remove_stop_words: bool = True) -> str:
    """Used to preprocess the pyrates book

    Args:
        text (str): Text to clean
        remove_stop_words (bool, optional): If you want to remove stop words. Defaults to True.

    Returns:
        str: cleaned text
    """

    # remove underscores
    text = text.replace("_", "")

    tokens = word_tokenize(text)

    if remove_stop_words:
        stop_words = stopwords.words("english")
        filtered_tokens = [
            word.lower()
            for word in tokens
            if word.lower() not in stop_words and word not in string.punctuation
        ]
    else:
        filtered_tokens = [
            word.lower() for word in tokens if word.lower() not in string.punctuation
        ]

    filtered_text = " ".join(filtered_tokens)

    # remove other special characters
    filtered_text = filtered_text.replace("'d", "")
    filtered_text = filtered_text.replace("'s", "")

    return filtered_text


def pre_process_rc(text: str, remove_stop_words: bool = True) -> str:
    """Used to preprocess Robinson Crusoe

    Args:
        text (str): Text to clean
        remove_stop_words (bool, optional): If you want to remove stop words. Defaults to True.

    Returns:
        str: cleaned text
    """

    # remove special punctuation
    text = text.replace("“", "")
    text = text.replace("”", "")
    text = text.replace("’", "")

    tokens = word_tokenize(text)

    if remove_stop_words:
        stop_words = stopwords.words("english")
        filtered_tokens = [
            word.lower()
            for word in tokens
            if word.lower() not in stop_words and word not in string.punctuation
        ]
    else:
        filtered_tokens = [
            word.lower() for word in tokens if word.lower() not in string.punctuation
        ]

    filtered_text = " ".join(filtered_tokens)

    return filtered_text


def pre_process_gt(text: str, remove_stop_words: bool = True) -> str:
    """Used to preprocess Gullivar's Travels

    Args:
        text (str): text to clean
        remove_stop_words (bool, optional): If you want to remove stop words. Defaults to True.

    Returns:
        str: cleaned text
    """

    # remove special punctuation
    text = text.replace("_", "")
    text = text.replace("“", "")
    text = text.replace("”", "")
    text = text.replace("’", "")

    tokens = word_tokenize(text)

    if remove_stop_words:
        stop_words = stopwords.words("english")
        filtered_tokens = [
            word.lower()
            for word in tokens
            if word.lower() not in stop_words and word not in string.punctuation
        ]
    else:
        filtered_tokens = [
            word.lower() for word in tokens if word.lower() not in string.punctuation
        ]

    filtered_text = " ".join(filtered_tokens)

    return filtered_text


def pre_process_weekly_journal(text: str) -> str:
    """Used to clean teh weekly journal

    Args:
        text (str): text to clean

    Returns:
        str: cleaned text
    """

    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = cleaned_text.split()
    valid_words = set(words.words())
    filtered_tokens = [word for word in tokens if word.lower() in valid_words]
    filtered_text = " ".join(filtered_tokens)

    return filtered_text


def determine_tf_idf_matrix(corpus: list) -> pd.DataFrame:
    """create the TF-IDF matrix using a more common method taught in courses

    Args:
        corpus (list): documents

    Returns:
        pd.DataFrame: TF IDF in dataframe form
    """

    vectorizer = TfidfVectorizer(
        input="content", smooth_idf=False, norm="l2", use_idf=False
    )
    x = vectorizer.fit_transform(corpus)

    n_samples, _ = x.shape
    array_sums = np.sum(x.toarray() > 0, axis=0)

    # TfidfVectorizer uses np.log(n_samples/array_sums) + 1
    idf_scale = np.log(n_samples / array_sums)
    tf_idf = x.toarray() * idf_scale

    return pd.DataFrame(tf_idf)


def determine_distance_matrix(tf_idf: pd.DataFrame) -> pd.DataFrame:
    """Used to create a distance matrix

    Args:
        tf_idf (pd.DataFrame): TF IDF matrix

    Returns:
        pd.DataFrame: distances between documents
    """
    return pd.DataFrame(euclidean_distances(tf_idf))
