import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from collections import Counter
import pickle

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000