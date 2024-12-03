# A General History of the Pyrates

This is an authorship identification project used for an NLP course. The following explains how to create your python environment and the directory structure. This is not intended for performance, rather a repo where a person new to building RNNs or other similar models could follow along and understand the process behind preparing, training, and uses a model.

## Python environment

The python environment is designed to be ran in a GPU environment. Although this is not necessary to train the RNN, it will take longer to build and train the model in a CPU environment.

To create the environment:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the `nltk` function, you will also need to download the following at the start, but will not need to replicate.

```
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
```

## Directory structure

All notebooks are found in [`notebooks`](./notebooks/). Within the directory there are the following:

* [Data prep](./notebooks/data_prep/) - used for preparing the data for both exploratory analysis and building the models.
* [Model data](./notebooks/model_data/`) - data you can use for the model
* [model_notebooks](./notebooks/model_notebooks/) - notebooks for both [TF-IDF](./notebooks/model_notebooks/tf_idf.ipynb) and for the [RNN](./notebooks/model_notebooks/build_model_with_sample.ipynb)

## Download GloVe embeddings

First you need to download, then you need to unzip. Make note of the full file path as you will need this to load the embeddings for your model.

```
wget https://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

### Loading embeddings

This function, found in [`notebooks/model_notebooks/utils/helpers.py`](https://github.com/bryaneadams/general_history_of_the_pyrates/blob/66095d300a982c4211bbd76a7e8e1081a3ffe740/notebooks/model_notebooks/utils/helpers.py#L198) will load embeddings into a dictionary for use in the model. 

There are a few values I skip for example:

`at name@domain.com` is skipped. This is ok, because it is not used in the analysis.

```
def load_glove_embeddings(file_path:str)->dict:
    """Loads the GloVe embeddings

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
```



## Book sources

1. [A General History of Pyrates](https://www.gutenberg.org/ebooks/search/?query=A+General+History+of+Pyrates&submit_search=Go%21)
2. [Daniel Defoe](https://www.gutenberg.org/ebooks/search/?query=Daniel+Defoe&submit_search=Go%21)
3. [Jonathan Swift](https://www.gutenberg.org/ebooks/author/326)
4. [Mist's Weekly Journal](https://go-gale-com.mutex.gmu.edu/ps/i.do?title=Mist%27s%2BWeekly%2BJournal&v=2.1&u=viva_gmu&it=JIourl&p=BBCN&sw=w)
