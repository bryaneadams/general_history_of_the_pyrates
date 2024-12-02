import string, re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

from typing import Tuple
from numpy.typing import ArrayLike

def determine_model_performance(model, test_data: ArrayLike, test_labels: ArrayLike) -> Tuple[ArrayLike,ArrayLike,float]:
    """
    Used to create a confusion matrix for your model
    
    Args:
        model (): Your model
        test_data (ArrayLike): Test data that is padded
        test_labels (ArrayLike): Test labels
        
    Returns:
        Tuple[ArrayLike,ArrayLike,float]: Confusion matrix, normalized confusion matrix, accuracy score
    """
    
    predictions = model.predict(test_data)
    prediction_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(test_labels, prediction_labels)
    score = accuracy_score(test_labels, prediction_labels)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm, cmn, score

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

def load_glove_embeddings(file_path:str)->dict:
    """_summary_

    Args:
        file_path (str): path to glove index

    Returns:
        dict: dictionary of glove embeddings
    """
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                pass
    return embeddings_index