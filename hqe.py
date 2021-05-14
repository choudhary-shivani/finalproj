# Historical Query Expansion Module for CIS.
# This module takes a list of utterances as input. Then we consider individual tokens in the utterances.
# For each token we then check BM25 retrieval score corresponding to most relevant document corresponding to the token.
# This BM25 score is obtained form pyserini and prebuilt CAST index.
# Finally, a turn ambiguity score is computed. If ambiguity is higher than a certain thershold, HQE is done.
from copy import deepcopy
from utils import preprocess_utterance


class HQE(object):
    def __init__(self, ir_engine, cfg):
        """
        :param ir_engine: PySerini Backend Engine over CAST documents.
        :param cfg: dict Configuration for cfg.
        """
        self.backend_engine = ir_engine
        self.cfg = cfg

        self.qt_thresh = cfg['qt_thresh']  # Query Token Threshold (R_q)
        self.st_thresh = cfg['st_thresh']  # Session Token Threshold (R_s)
        self.q_thresh = cfg['q_thresh']    # Query Threshold (Theta)
        self.last_k = cfg['last_k']        # Last K query keywords to append when current query is ambigious

        self.use_orig_for_query = cfg.get('use_orig_for_query', True)

        print(f'Loaded HQE module.')
        print(f'use_orig_for_query: {self.use_orig_for_query}')

    def expand_queries(self, utterances):
        """
        :param utterances: [str] list of utterances
        :return expanded_sets: [str] an expanded query set
        """
        expanded_sets = []
        proc_uttrs = [preprocess_utterance(ctx, remove_stopwords=True,
                                           lower=True
                                           ) for ctx in utterances]

        # Seperate expansion tokens at Query and Session Level. Does not make a difference if we use same thresholds.
        W_Q = [set() for _ in range(self.last_k)]
        W_S = set()

        for uid, uttr in enumerate(proc_uttrs):
            tokens = uttr.split()
            # temp = []
            # for token_ in tokens:
            #     if all([i.isalpha() for i in token_]) and len(token_) > 2:
            #         temp.append(token_)
            # tokens = temp
            # This means we are causing trouble in preprocessing. Removing stopwords may not be a good idea.
            # print(uttr, tokens)
            assert len(tokens) != 0

            # Update token selections based on the BM25 score
            q_set = set()
            for tok in tokens:
                results = self.backend_engine.search(tok, k=1)  # Only best hit matters

                if len(results) > 0:
                    # Add the tokens to W_Q and W_S
                    match_score = results[0].score
                    
                    if match_score > self.qt_thresh:
                        q_set.add(tok)
 
                    if match_score > self.st_thresh:
                        W_S.add(tok)

            # Now perform query expansion. Note that we donot expand first query.
            if uid == 0:
                W_Q[uid % self.last_k] = q_set
                expanded_sets.append(set())
                continue

            # We are using original query here. Note that this makes sense only when we are using manual runs
            # For example utterance like 'why is that the case' is not at all informative. But, 'why does water vapourizes at 100 degrees?'
            # is of more sense. This basically restricts model to manual runs only. Author's use this for automatic evaluation.
            
            if self.use_orig_for_query:
                results = self.backend_engine.search(utterances[uid], k=1)
            else:
                results = self.backend_engine.search(uttr, k=1)

            if len(results) == 0:
                # Very bad query this is. Expand this we will.
                print(f'Anomaly detected: No documents found for query {utterances[uid]}')
                match_score = -1
            else:
                match_score = results[0].score

            extension_set = deepcopy(W_S)

            if match_score < self.q_thresh:
                # Ambiguious results. Add context from last k queries.
                query_set = set().union(*W_Q)
                extension_set = extension_set.union(query_set)
            # print(extension_set)
            expanded_sets.append(extension_set)
            # print(expanded_sets)
            # print ("--------------------------")
            W_Q[uid % self.last_k] = q_set

        return expanded_sets
