# Passage Query Expansion Module.
# Given a query we do the followin
# 1. First we classify utterance as implicit and explicit
# 2. For explicit utterances we expand the query
#
# For explicit queries, get top-k documents to get expanded keywords.

import gc, joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import preprocess_utterance as preprocess_utterance


class PQE(object):
    def __init__(self, ir_engine, vectorizer, top_k):
        """
        :param ir_engine: PySerini Backend Engine over CAST documents.
        :param vectorizer: TfidfVectorizer vectorizer
        :param cfg: dict Configuration for cfg.
        """
        self.backend_engine = ir_engine
        self.vectorizer = vectorizer
        self.top_k = top_k
        self.feature_names = self.vectorizer.get_feature_names()

    def classify_utterance(self, utterance):
        """
        :param utterance: str
        :return: True if utterance to be expanded else False
        """
        return True

    def expand_query(self, utterance):
        """
        :param utterance: str
        :return query: str expanded query 
        """
        if not self.classify_utterance(utterance):
            return utterance

        # 1. Get top-k documents
        results = self.backend_engine(utterance, k=self.top_k)

        if len(results) < self.top_k:
            print(f'Number of results {len(results)} are less than top-k {self.top_k}.')
            print(f'Query: {utterance}')

        if len(results) == 0:
            return utterance

        documents = [res.raw for res in results]

        # 2. Get TF-IDF query expansion
        tfidf_scores = self.vectorizer.transform(documents)  # doc_num x vocabsize
        sort_idxs = np.argsort(tfidf_scores.data)[::-1][:self.top_k]
        vocab_idxs = tfidf_scores.indices[sort_idxs]
        extension_set = set([self.feature_names[idx] for idx in vocab_idxs])

        return utterance + " " + " ".join(extension_set)
        

def dump_tf_idf_msmarco(fname, dest):
    print(f'Preparing TF IDF featurizer for MSMARCO')
    print(f'Reading data from {fname}')
    df = pd.read_csv(
        fname, header=None,
        names=['did', 'document'], sep='\t',
    )
    print(f'Number of rows: {df.shape[0]}')

    df.drop_duplicates(subset=['did', 'document'], inplace=True)
    print(f'Number of rows after deduplication: {df.shape[0]}')

    documents = df.document.tolist()
    del df
    gc.collect()

    print(documents[:10])
    vectorizer = TfidfVectorizer(
        stop_words='english',
        preprocessor=preprocess_utterance,
        strip_accents='ascii', lowercase=True,
    ).fit(documents)
    vectorizer.fit(documents)

    joblib.dump(vectorizer, dest)


if __name__ == '__main__':
    dest = '/mnt/workdrive/Study/CAST/msmarco/msmarco_tfidf.pkl'
    fname = '/mnt/workdrive/Study/CAST/msmarco/collection.tsv'

    dump_tf_idf_msmarco(fname, dest)
