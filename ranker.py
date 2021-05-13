import os
import torch
import numpy as np
torch.set_num_threads(os.cpu_count()-1)

class rerank:
    def __init__(self, ir_engine, passage_max_len, query_max_len, tokenizer,
                 model, verbose=False, list_of_files=None, query=None):
        self.query = query
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
        self.list_of_files = list_of_files
        self.tokenizer = tokenizer
        self.model = model
        self.searcher = ir_engine
        self.verbose = verbose

    def _query_passage_rerank(self, doc_id):
        passage = self.searcher.doc(doc_id).raw()
        q_tok = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('[CLS]' + self.query + '[SEP]'))
        p_tok = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(passage + '[SEP]'))
        # Check the limitation on query and passage
        if len(q_tok) > self.query_max_len:
            q_tok = q_tok[0] + q_tok[1: self.query_max_len-1] + q_tok[1]
        if len(q_tok) > self.passage_max_len:
            p_tok = q_tok[0] + p_tok[1: self.passage_max_len-1] + p_tok[1]
        s_p_id = [1] * len(p_tok)
        s_q_id = [0] * len(q_tok)
        if len(q_tok + p_tok) < 512:
            pad = ['[PAD]'] * (512 - len(q_tok + p_tok))
            pad_tok = self.tokenizer.convert_tokens_to_ids(pad)
            seq_pad = [1] * (512 - len(q_tok + p_tok))
            t_q_tok = torch.tensor(q_tok + p_tok + pad_tok)
            seq_id_final = torch.tensor(s_q_id + s_p_id + seq_pad)
            return t_q_tok, seq_id_final
        t_q_tok = torch.tensor([q_tok + p_tok])
        seq_id_final = torch.tensor([s_q_id + s_p_id])
        return [t_q_tok, seq_id_final]

    def rerank(self, query, doc_list):
        assert isinstance(doc_list, list)  # ensure that we get a list of
        # file names
        self.query = query
        self.list_of_files = doc_list
        seq = []
        tok_id = []
        for i in self.list_of_files:
            f_list = self._query_passage_rerank(i)
            tok_id.append(f_list[0])
            seq.append(f_list[1])
        with torch.no_grad():
            score = self.model(torch.stack(tok_id, dim=0), torch.stack(seq,
                                                                     dim=0))
        # Sort the array based on the classfication score and collecting the
        # index so that we can rerank
        y = np.argsort(score.logits.detach().numpy().ravel()[::2])
        if self.verbose:
            print("New sequence is {}".format(np.array(self.list_of_files)[y]))
        return list(np.array(self.list_of_files)[y])



if __name__ == '__main__':
    re = rerank('What is a physician\u0027s assistant?',
                ['MARCO_5579926', 'MARCO_4691210', 'MARCO_6381528', 'MARCO_5579927', 'MARCO_1183715',
                 'MARCO_3270013', 'MARCO_5579925', 'MARCO_3929687', 'MARCO_1865839', 'MARCO_6381530'])
    re.rerank()