# A General History of the Pyrates

Used for a course project in my NLP course.

## Python environment

```
import nltk
nltk.download('stopwords')
nltk.download('words')
```

## Download GloVe embeddings

First you need to download, then you need to unzip. Make note of the full file path as you will need this to load the embeddings for your model.

```
wget https://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

### Loading embeddings

This function will load embeddings into a dictionary for use in the model. There are a few values I skip for example:

`at name@domain.com` is skipped. This is ok, because it is not used in the analysis.

```
def load_glove_embeddings(file_path):
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