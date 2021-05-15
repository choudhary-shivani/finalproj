import sys
sys.path.append('/home/vsaley/additional_dependancies')

import numpy as np
from time import time
from collections import Counter
import string, joblib, unicodedata

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = stopwords.words('english')
stop_words.append('would')
lemmatizer = WordNetLemmatizer()

# from trec_car import read_data


def text_processor(text):
    """
    Preprocess a single utterance in CAST data.
    """
    text = text.lower()

    # Normalize accents
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")

    # Remove punctuations
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove stop words
    tokens = filter(lambda x: x not in stop_words, text.split())

    # Lemmatize
    text = " ".join(
        [lemmatizer.lemmatize(x) for x in tokens]
    )

    return text


def get_msmarco_documents(fname, idf_counter):
    print(f'Reading MSMARCO documents from {fname}.')

    st = time()
    with open(fname, 'r') as fp:
        doc_cnt = 0
        for _, line in enumerate(fp):
            doc = line.split()[-1].strip()
            doc = text_processor(doc)
            idf_counter.update(doc.split())

            doc_cnt += 1

    en = time()
    print(f'Processed {doc_cnt} from MSMARCO in {en - st}s')

    return idf_counter


def get_car_documents(fname, idf_counter):
    doc_cnt = 0

    st = time()
    for _, para in enumerate(read_data.iter_paragraphs(open(fname, 'rb'))):
        doc = para.get_text()
        doc = text_processor(doc)
        idf_counter.update(doc.split())

        doc_cnt += 1
        if doc_cnt % 1000000 == 0:
            print(f'Completed {doc_cnt} CAR read at {time() - st}s')

    en = time()
    print(f'Processed {doc_cnt} from CAR in {en - st}s')

    return idf_counter


def build_idf_counter(msmarco_loc, car_loc, dest):
    idf_counter = Counter()
    idf_counter = get_msmarco_documents(msmarco_loc, idf_counter)
    idf_counter = get_car_documents(car_loc, idf_counter)

    print(f'IDF Counter at {dest}')
    joblib.dump(idf_counter, dest)


def process_idf_counter(fname):
    idf_counter = joblib.load(fname)
    print(f'Vocab Counts: {idf_counter}')
    total_doc_cnt = 38636520
    idf_values = {}

    numer = total_doc_cnt + 1
    for k, v in idf_counter.items():
        denom = v + 1
        idf_values[k] = np.log(numer / denom)

    joblib.dump(idf_values, fname)


if __name__ == '__main__':
    msmarco_loc = '/home/vsaley/Courses/TNLP/data/msmarco/collection.tsv'
    car_loc = '/home/vsaley/Courses/TNLP/data/CAR/paragraphCorpus/dedup.articles-paragraphs.cbor'
    dest = '/home/vsaley/Courses/TNLP/data/idf_lemmatized_counter.pkl'

    # build_idf_counter(msmarco_loc, car_loc, dest)
    process_idf_counter(dest)