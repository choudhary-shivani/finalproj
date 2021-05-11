import os, joblib
import numpy as np

from utils import read_topics_as_utterances
from pyserini.search import SimpleSearcher

from hqe import HQE
from pqe import PQE

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
cast_index_loc = '/mnt/workdrive/Study/CAST/indices/index-cast2019'
cast_data_loc = '/mnt/workdrive/Study/CAST/treccastweb/2020/2020_automatic_evaluation_topics_v1.0.json'
tfidf_loc = '/mnt/workdrive/Study/CAST/idf_counter.pkl'

# Data Related
training_topics = '/mnt/workdrive/Study/CAST/treccastweb/2019/data/training/train_topics_v1.0.json'
evaluation_topics = '/mnt/workdrive/Study/CAST/treccastweb/2019/data/evaluation/evaluation_topics_v1.0.json'

config = {
    'cardinality': 10,
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
    }
}

class Pipeline(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.cardinality = cfg['cardinality']

        self.backend_engine = None
        self.idf = None
        self.expansion_pipeline = None

        self.prepare_pipeline(cfg)

    def prepare_pipeline(self, cfg):
        print(f'Loading Backed PySerini Engine from {cast_index_loc}')
        self.backend_engine = SimpleSearcher(cast_index_loc)

        print(f'Loading IDF values from {tfidf_loc}')
        self.idf = joblib.load(tfidf_loc)

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

        self.expansion_pipeline = components
        
    def query_expansion(self, utterances):
        if len(self.expansion_pipeline) == 0:
            return utterances

        for module in self.expansion_pipeline:
            utterances = module.expand_queries(utterances)

        return utterances

    def query_execution(self, utterances):
        results = []
        
        for query in utterances:
            hits = self.backend_engine.search(query)
            results.append(
                [hit.docid for hit in hits]
            )
        
        return results

    def execute(self, utterances):
        queries = self.query_expansion(utterances)
        results = self.query_execution(queries)

        return results


if __name__ == '__main__':
    pipeline = Pipeline(config)
    train_utterances = read_topics_as_utterances(training_topics)[:1]
    ret = pipeline.execute(train_utterances[0])

    print(ret)