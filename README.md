# A General History of the Pyrates

Used for a course project in my NLP course.

## Python environment

```
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
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
### Book sources

1. [A General History of Pyrates](https://www.gutenberg.org/ebooks/search/?query=A+General+History+of+Pyrates&submit_search=Go%21)
2. [Daniel Defoe](https://www.gutenberg.org/ebooks/search/?query=Daniel+Defoe&submit_search=Go%21)
3. [Jonathan Swift](https://www.gutenberg.org/ebooks/author/326)
4. [Mist's Weekly Journal](https://go-gale-com.mutex.gmu.edu/ps/i.do?title=Mist%27s%2BWeekly%2BJournal&v=2.1&u=viva_gmu&it=JIourl&p=BBCN&sw=w)