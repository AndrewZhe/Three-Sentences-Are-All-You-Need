"""
GCN model for relation extraction.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model.BiLSTM import EncoderLSTM
from model.localEncoder import LocalEncoder
from model.classifier import EntityClassifier


class DocClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        # Doc Embedding 
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=0)
        self.init_embeddings(emb_matrix)
        self.ner_embed = nn.Embedding(opt['ner_num'], opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.coref_embed = nn.Embedding(opt['max_len'], opt['coref_dim']) if opt['coref_dim'] > 0 else None
        self.in_drop = nn.Dropout(opt['input_dropout'])
        
        # Global Encoder
        in_dim = opt['emb_dim'] + opt['ner_dim'] + opt['coref_dim']
        self.global_encoder = EncoderLSTM(in_dim, opt['hidden_dim'] // 2, 1, True, True, opt['rnn_dropout'], False, True)
        self.rnn_drop = nn.Dropout(opt['rnn_dropout'])
        
        self.entity_classifier = EntityClassifier(opt['hidden_dim'], opt['num_class'], opt['mlp_dropout'])

        # Local Encoder
        self.local_encoder = LocalEncoder(in_dim, opt['hidden_dim'], opt['input_dropout'], opt['rnn_dropout'],
                                          self.emb, self.ner_embed, self.coref_embed, opt['max_len'], opt['pos_dim'])


    def init_embeddings(self, emb_matrix):
        # 决定word embedding中的哪些部分更新，但实际上只有UNK会更新
        def keep_partial_grad(grad, topk):
            """
            Keep only the topk rows of grads.
            """
            assert topk < grad.size(0)
            grad.data[topk:].zero_()
            return grad
        
        if emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            emb_matrix = torch.from_numpy(emb_matrix)
            self.emb.weight.data.copy_(emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")
    
    def forward(self, sample):
        """
          words   : (bz, doc_len) int64
          ner     : (bz, doc_len) int64
          coref   : (bz, doc_len) int64
          length  : (bz) int64
          head_mask: (bz, doc_len) float32
          tail_mask: (bz, doc_len) float32

          sent_doc_mp: (bz, sent_num, doc_len) float32
          sent_num  : (bz, ) int64
          support_set: (bz, sent_num)  #float32
        """
        for k, v in sample.items():
            sample[k] = v.cuda()
        words, ner, coref, length, head_mask, tail_mask = \
            sample['words_id'], sample['ners_id'], sample['coref_id'], sample['doc_len'], \
            sample['head_mask'], sample['tail_mask']

        # Doc Embedding
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['ner_dim'] > 0:
            embs += [self.ner_embed(ner)]
        if self.opt['coref_dim'] > 0:
            embs += [self.coref_embed(coref)]
        
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        doc_outputs, last_h = self.global_encoder(embs, length)

        doc_outputs = self.rnn_drop(doc_outputs)  # (bz, doc_len, hs)
        doc_outputs = self.rnn_drop(doc_outputs)  # (bz, doc_len, hs)

        head_mask.unsqueeze_(1)
        tail_mask.unsqueeze_(1)
        global_head = torch.matmul(head_mask, doc_outputs).squeeze(1)  # (bz, hs)
        global_tail = torch.matmul(tail_mask, doc_outputs).squeeze(1)  # (bz, hs)

        # Document Representation
        # Local Encoder
        local_head, local_tail = self.local_encoder(sample, doc_outputs, embs, head_mask, tail_mask)
        
        # Classifier
        pred = self.entity_classifier(global_head, global_tail, local_head, local_tail, sample['path2ins'])
        
        return pred


