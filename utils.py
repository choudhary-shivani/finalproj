import string

import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append('would')


def preprocess_utterance(utterance, remove_stopwords=False):
    """
    Preprocess a single utterance in CAST data.
    """
    utterance = utterance.lower()
    ret = utterance.translate(str.maketrans('', '', string.punctuation))

    if remove_stopwords:
        ret = " ".join(
            filter(lambda x: x not in stop_words, ret.split())
        )

    return ret
