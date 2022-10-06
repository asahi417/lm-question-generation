"""https://github.com/AIPHES/emnlp19-moverscore/blob/b9d4144e785293849216618e85db887b3ac741d3/moverscore_v2.py"""
import os
import string
from collections import defaultdict
from typing import List
from tqdm import tqdm

import torch
import numpy as np
from pyemd import emd_with_flow
from transformers import AutoTokenizer, AutoModel

MOVERSCORE_BATCH = int(os.getenv('MOVERSCORE_BATCH', '256'))
# lang2model = defaultdict(lambda: "bert-base-multilingual-cased")
# lang2model.update(
#     {
#         "en": "roberta-large",
#         "zh": "bert-base-chinese",
#         "tr": "dbmdz/bert-base-turkish-cased",
#         "en-sci": "allenai/scibert_scivocab_uncased",
#     }
# )

lang2model = defaultdict(lambda: "bert-base-multilingual-cased")
lang2model.update({"en": "distilbert-base-uncased"})


def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
        x1_norm).clamp_min_(1e-30).sqrt_()
    return res


class MoverScore:

    def __init__(self, language: str = 'en'):
        model_name = lang2model[language]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        except ValueError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, local_files_only=True)
        try:
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        except ValueError:
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True,
                                                   local_files_only=True)
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def collate_idf(self, arr, idf_dict, pad="[PAD]"):

        def padding(_arr, dtype=torch.long):
            lens = torch.LongTensor([len(a) for a in _arr])
            max_len = lens.max().item()
            padded = torch.ones(len(_arr), max_len, dtype=dtype) * self.tokenizer.convert_tokens_to_ids([pad])[0]
            mask = torch.zeros(len(_arr), max_len, dtype=torch.long)
            for i, a in enumerate(_arr):
                padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
                mask[i, :lens[i]] = 1
            return padded, lens, mask

        def truncate(_tokens):
            if len(_tokens) > self.tokenizer.model_max_length - 2:
                return _tokens[0:(self.tokenizer.model_max_length - 2)]
            return _tokens

        tokens = [["[CLS]"] + truncate(self.tokenizer.tokenize(a)) + ["[SEP]"] for a in arr]
        arr = [self.tokenizer.convert_tokens_to_ids(a) for a in tokens]
        idf_weights = [[idf_dict[i] for i in a] for a in arr]
        _padded, _lens, _mask = padding(arr, dtype=torch.long)
        padded_idf, _, _ = padding(idf_weights, dtype=torch.float)

        return _padded, padded_idf, _lens, _mask, tokens

    def get_bert_embedding(self, all_sens, idf_dict, batch_size=-1):

        def bert_encode(x, attention_mask):
            result = self.model(x.to(self.device), attention_mask=attention_mask.to(self.device))
            return result[1] if self.model_name == 'distilbert-base-uncased' else result[2]

        padded_sens, padded_idf, lens, mask, tokens = self.collate_idf(all_sens, idf_dict)

        batch_size = len(all_sens) if batch_size == -1 else batch_size
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_sens), batch_size):
                batch_embedding = bert_encode(padded_sens[i:i + batch_size], attention_mask=mask[i:i + batch_size])
                batch_embedding = torch.stack(batch_embedding).cpu()
                embeddings.append(batch_embedding)
                del batch_embedding

        total_embedding = torch.cat(embeddings, dim=-3)
        return total_embedding, lens, mask, padded_idf, tokens

    def word_mover_score(self, refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=None, batch_size=256):
        stop_words = [] if stop_words is None else stop_words
        preds = []
        for batch_start in tqdm(list(range(0, len(refs), batch_size))):
            batch_refs = refs[batch_start:batch_start + batch_size]
            batch_hyps = hyps[batch_start:batch_start + batch_size]

            ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = self.get_bert_embedding(batch_refs, idf_dict_ref)
            hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = self.get_bert_embedding(batch_hyps, idf_dict_hyp)

            ref_embedding = ref_embedding[-1]
            hyp_embedding = hyp_embedding[-1]

            batch_size = len(ref_tokens)
            for i in range(batch_size):
                ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                           w in stop_words or '##' in w or w in set(string.punctuation)]
                hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                           w in stop_words or '##' in w or w in set(string.punctuation)]

                ref_embedding[i, ref_ids, :] = 0
                hyp_embedding[i, hyp_ids, :] = 0

                ref_idf[i, ref_ids] = 0
                hyp_idf[i, hyp_ids] = 0

            raw = torch.cat([ref_embedding, hyp_embedding], 1)
            raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30)
            distance_matrix = batched_cdist_l2(raw, raw).double().cpu().numpy()

            for i in range(batch_size):
                c1 = np.zeros(raw.shape[1], dtype=float)
                c2 = np.zeros(raw.shape[1], dtype=float)
                c1[:len(ref_idf[i])] = ref_idf[i]
                c2[len(ref_idf[i]):] = hyp_idf[i]
                c1 = c1 / (np.sum(c1) + 1e-30)
                c2 = c2 / (np.sum(c2) + 1e-30)
                dst = distance_matrix[i]
                _, flow = emd_with_flow(c1, c2, dst)
                flow = np.array(flow, dtype=np.float32)
                score = 1. / (1. + np.sum(flow * dst))  # 1 - np.sum(flow * dst)
                preds.append(score)

        return preds

    def get_score(self, hypothesis: List, references: List):
        assert len(hypothesis) == len(references)
        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)
        return self.word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, batch_size=MOVERSCORE_BATCH)

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        _ids = gts.keys()
        hyps = [res[_id][0].decode() for _id in _ids]
        refs = [gts[_id][0].decode() for _id in _ids]
        _score = self.get_score(hyps, refs)
        average_score = np.mean(_score)
        return average_score, np.array(_score)

    @staticmethod
    def method():
        return "MoverScore"
