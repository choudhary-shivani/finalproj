# Passage Query Expansion Module.
# Given a query we do the followin
# 1. First we classify utterance as implicit and explicit
# 2. For explicit utterances we expand the query
#
# For explicit queries, get top-k documents to get expanded keywords.

import gc, joblib
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import spacy

from tf_idf_vectorizer import text_processor as idf_processor
from utils import preprocess_utterance as preprocess_utterance


class PQE(object):
    def __init__(self, ir_engine, idf, top_k_documents, top_k_tokens):
        """
        :param ir_engine: PySerini Backend Engine over CAST documents.
        :param idf: Token Inverse document frequency
        :param cfg: dict Configuration for cfg.
        """
        self.backend_engine = ir_engine
        self.idf = idf
        self.top_k_documents = top_k_documents
        self.top_k_tokens = top_k_tokens
        
        self.nlp = spacy.load("en_core_web_sm")
        self.ignore_pronoun_list = [
            'i', 'me', 'my', 'myself', 'you', 'your',
            # 'why', 'who', 'when', 'where'
        ]

    def classify_utterance(self, utterance):
        """
        :param utterance: str
        :return: True if utterance to be expanded else False
        """
        doc = self.nlp(utterance)
        res = any([
            (x.text not in self.ignore_pronoun_list) and (x.pos_ == 'PRON')
            for x in doc
        ])

        return res

    def get_topk_token(self, documents):
        scores = defaultdict(lambda : -1.0)

        for doc in documents:
            tokens = idf_processor(doc).split()
            tf = Counter(tokens)

            for tok in tf:
                score = tf[tok] * self.idf.get(tok, 0)
                if score > scores[tok]:
                    scores[tok] = score

        sorted_items = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )[:self.top_k_tokens]

        return set([x[0] for x in sorted_items])

    def expand_query(self, utterance):
        """
        :param utterance: str
        :return query: str expanded query 
        """
        if not self.classify_utterance(utterance):
            return utterance

        # 1. Get top-k documents
        results = self.backend_engine.search(utterance, k=self.top_k_documents)

        if len(results) < self.top_k_documents:
            print(f'Number of results {len(results)} are less than top-k {self.top_k_documents}.')
            print(f'Query: {utterance}')

        if len(results) == 0:
            return utterance

        documents = [res.raw for res in results]

        # 2. Get TF-IDF query expansion
        extension_set = self.get_topk_token(documents)

        return extension_set

    def expand_queries(self, utterances):
        return [
            self.expand_query(uttr) for uttr in utterances
        ]