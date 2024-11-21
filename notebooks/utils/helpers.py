import string, re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer


def pre_process_pyrates(text: str, remove_stop_words: bool = True) -> str:
    """_summary_

    Args:
        text (str): _description_
        remove_stop_words (bool, optional): _description_. Defaults to True.

    Returns:
        str: _description_
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
    """_summary_

    Args:
        text (str): _description_
        remove_stop_words (bool, optional): _description_. Defaults to True.

    Returns:
        str: _description_
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
    """_summary_

    Args:
        text (str): _description_
        remove_stop_words (bool, optional): _description_. Defaults to True.

    Returns:
        str: _description_
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
    """_summary_

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """

    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = cleaned_text.split()
    valid_words = set(words.words())
    filtered_tokens = [word for word in tokens if word.lower() in valid_words]
    filtered_text = " ".join(filtered_tokens)

    return filtered_text


def determine_tf_idf_matrix(corpus: list) -> pd.DataFrame:
    """_summary_

    Args:
        corpus (list): _description_

    Returns:
        pd.DataFrame: _description_
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
    """_summary_

    Args:
        tf_idf (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return pd.DataFrame(euclidean_distances(tf_idf))
