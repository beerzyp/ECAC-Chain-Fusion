import nltk
from nltk.corpus import stopwords
import pandas as pd
import string
from collections import Counter
from keras.preprocessing.text import Tokenizer
import random
nltk.download('stopwords')

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

if __name__ == "__main__":
    # load the vocabulary
    vocab_filename = 'data/vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    data = pd.read_csv('data/styles.csv', error_bad_lines=False)

    sentences = data['productDisplayName'].values.tolist()
    for sentence in sentences:
        if not isinstance(sentence, str):
            sentences.remove(sentence)

    # create the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    
    #TODO: Split into two in a even way
    random.shuffle(sentences)

    #construct train and test data
    split_num = int(len(sentences) * 0.7)
    train_data = sentences[:split_num]
    test_data = sentences[split_num:]

    X_train = tokenizer.texts_to_matrix(train_data, mode='freq')
    print(X_train.shape)

    X_test = tokenizer.texts_to_matrix(test_data, mode='freq')
    print(X_test.shape)