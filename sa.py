import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from collections import Counter
import pickle

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r')as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_count = Counter(lexicon)
    l2 = []
    for w in w_count:
        if 1000 > w_count(w) > 50:
            l2.append(w)
        return l2


def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
                    features = list(features)
                    featureset.append([[featureset, classification]])
    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon[1, 0])
    features += sample_handling('neg.txt', lexicon[0, 1])
    random.shuffle(features)

    features = np.array(features)
    test_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-test_size])
    train_y = list(features[:, 1][:-test_size])

    test_x = list(features[:, 0][-test_size:])
    test_y = list(features[:, 1][-test_size:])
