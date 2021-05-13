import gc
import time
import os, joblib
import _pickle as cpickle
import numpy as np

from utils import read_topics_as_utterances
from pyserini.search import SimpleSearcher
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from hqe import HQE
from pqe import PQE
from ranker import rerank

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
cast_index_loc = '/home/shivani/Downloads/index-cast2019.tar.gz/index-cast2019'
cast_data_loc = './treccastweb/2020/2020_automatic_evaluation_topics_v1.0.json'
tfidf_loc = '/home/shivani/Downloads/idf_counter.pkl'

# Data Related
training_topics = './treccastweb/2019/data/training/train_topics_v1.0.json'
evaluation_topics = './treccastweb/2019/data/evaluation/evaluation_topics_v1' \
                    '.0.json'

config = {
    'cardinality': 10,
    'verbose': False,
    'hqe': {
        'qt_thresh': 4,
        'st_thresh': 3,
        'q_thresh': 16,
        'last_k': 3,
        'use_orig_for_query': True
    },
    'pqe': {
        'top_k_documents': 100,
        'top_k_tokens': 5,
    },
    'ranker':
        {
            'passage_max_len': 448,
            'query_max_len': 64
        }
}


class Pipeline(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.cardinality = cfg['cardinality']

        self.backend_engine = None
        self.idf = None
        self.expansion_pipeline = None
        self.tokenizer = None
        self.model = None
        self.ranker_pipeline = None
        self.prepare_pipeline(cfg)

    def prepare_pipeline(self, cfg):

        print(f'Loading Backed PySerini Engine from {cast_index_loc}')
        self.backend_engine = SimpleSearcher(cast_index_loc)

        print(f'Loading IDF values from {tfidf_loc}')
        start = time.time()
        gc.disable()
        # self.idf = joblib.load(tfidf_loc)
        with open(tfidf_loc, 'rb') as f:
            self.idf = cpickle.load(f)
        gc.enable()
        print ("Loading completed in {} sec.".format(time.time() - start))
        print(f'Loading BERT reranker and tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(
            "amberoad/bert-multilingual-passage-reranking-msmarco")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "amberoad/bert-multilingual-passage-reranking-msmarco")

        components = []
        if 'hqe' in cfg:
            print(f'Adding HQE module to pipeline')
            hqe_cfg = cfg['hqe']
            hqe = HQE(self.backend_engine, hqe_cfg)
            components.append(hqe)

        if 'pqe' in cfg:
            print(f'Adding PQE module to pipeline')
            pqe_cfg = cfg['pqe']
            pqe = PQE(
                ir_engine=self.backend_engine,
                idf=self.idf,
                top_k_documents=pqe_cfg['top_k_documents'],
                top_k_tokens=pqe_cfg['top_k_tokens']
            )
            components.append(pqe)

        if 'ranker' in cfg:
            print(f'Adding Reranker module to pipeline')
            ranker_cfg = cfg['ranker']
            ranker = rerank(
                ir_engine=self.backend_engine,
                passage_max_len=ranker_cfg['passage_max_len'],
                query_max_len=ranker_cfg['query_max_len'],
                tokenizer=self.tokenizer,
                model=self.model,
                verbose=cfg['verbose']
            )
            self.ranker_pipeline = ranker

        self.expansion_pipeline = components
        print('Pipeline load completed.')

    def query_expansion(self, utterances):
        if len(self.expansion_pipeline) == 0:
            return utterances

        expanded_queries = utterances
        for module in self.expansion_pipeline:
            extension_sets = module.expand_queries(expanded_queries)
            expanded_queries = [
                utterances[idx] + " " + " ".join(extension_sets[idx])
                for idx in range(len(utterances))
            ]

        return expanded_queries

    def query_execution(self, utterances):
        results = []

        for query in utterances:
            hits = self.backend_engine.search(query)
            results.append(
                [hit.docid for hit in hits]
            )
        return results

    def passage_reranker(self, utterances, results):
        reranked_output = []
        for query, result in zip(utterances, results):
            reranked_output.append(self.ranker_pipeline.rerank(query, result))
        return reranked_output

    def execute(self, utterances):
        queries = self.query_expansion(utterances)
        results = self.query_execution(queries)
        reranked = self.passage_reranker(queries, results)
        return results, reranked


if __name__ == '__main__':
    pipeline = Pipeline(config)
    train_utterances = read_topics_as_utterances(training_topics)[:1]
    res, rerank = pipeline.execute(train_utterances[0])
    print("Non-reranked {}\nReranked output {}".format(res, rerank))
