import os
import torch
import numpy as np
from torch.nn.functional import softmax

torch.set_num_threads(os.cpu_count())


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
            q_tok = [q_tok[0]] + q_tok[1: self.query_max_len - 2] + \
                    [q_tok[-1]]
        if len(p_tok) > self.passage_max_len:
            p_tok = p_tok[:self.passage_max_len - 1] + [p_tok[-1]]

        s_p_id = [1] * len(p_tok)
        s_q_id = [0] * len(q_tok)

        # This is applied so that a batch of tensor can be created. It needs
        # to be of same length. Hence the PAD token is appended as input.

        if len(q_tok + p_tok) < 512:
            pad = ['[PAD]'] * (512 - len(q_tok + p_tok))
            pad_tok = self.tokenizer.convert_tokens_to_ids(pad)
            seq_pad = [1] * (512 - len(q_tok + p_tok ) )
            # print(q_tok + p_tok + pad_tok)
            t_q_tok = torch.tensor(q_tok + p_tok + pad_tok)
            seq_id_final = torch.tensor(s_q_id + s_p_id + seq_pad)
            attn_mask = torch.ByteTensor(([1] * (len(p_tok) + len(q_tok))) +
                                         [0] *(512 - (len(p_tok) + len(q_tok))))
            attention = torch.FloatTensor([1] * 512)
            # print(attention.shape, attn_mask.shape, len(p_tok), len(q_tok))
            attention.masked_fill_(attn_mask == 0, -np.inf)
            return t_q_tok, seq_id_final, attention

        t_q_tok = torch.tensor(q_tok + p_tok)
        seq_id_final = torch.tensor(s_q_id + s_p_id)
        return t_q_tok, seq_id_final, torch.FloatTensor([1]*512)

    def rerank(self, query, doc_list):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert isinstance(doc_list, list)  # expects doc list as input
        self.query = query
        self.list_of_files = doc_list
        seq = []
        tok_id = []
        attention = []
        self.model = self.model.to(device)

        for i in self.list_of_files:
            f_list = self._query_passage_rerank(i)
            # print(self.tokenizer.decode(f_list[0]))
            tok_id.append(f_list[0])
            seq.append(f_list[1])
            attention.append(f_list[2])

        with torch.no_grad():
            score = self.model.forward(torch.stack(tok_id, dim=0).to(device),
                               token_type_ids=torch.stack(seq, dim=0).to(
                                   device),
                               attention_mask=torch.stack(attention,
                                                      dim=0).to(device))

        # Sort the array based on the classfication score and collecting the
        # index so that we can rerank
        new_val = softmax(score.logits, dim=1)
        nt = new_val.cpu()
        final_tensor = nt.detach().numpy().ravel()[1::2]
        y = np.argsort(final_tensor)[::-1]
        # if self.verbose:
        #     print("New sequence is {} \n {}".format(np.array(
        #         self.list_of_files)[y], self.list_of_files))
        return list(np.array(self.list_of_files)[y])
