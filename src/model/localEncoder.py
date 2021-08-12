# -*- coding:utf-8 -*-
# Author: Quzhe Huang
# Date:   Nov. 13, 2020

import torch
import torch.nn as nn
from model.BiLSTM import EncoderLSTM


class LocalEncoder(nn.Module):
    def __init__(self, emb_dim, hs, indp, rdp, word_embed, ner_embed, coref_embed, max_len, pos_dim):
        """
        :param encoder_type: bilstm和raw_bilstm 都是把head 和 tail所在的所有句子concat起来，
            其中raw_bilstm是直接拿词向量作为输入，而bilstm是拿第一次层bilstm的输出作为输入。
            path_bilstm则是把每个entity的所有path单独编码，直接拿词向量作为输入。
        :param emb_dim:
        :param hs:
        :param dropout:
        """
        super().__init__()
        self.max_len = max_len

        self.word_embed = word_embed
        self.ner_embed = ner_embed
        self.coref_embed = coref_embed
        self.head_pos_embed = nn.Embedding(2*max_len, pos_dim) if pos_dim > 0 else None
        self.tail_pos_embed = nn.Embedding(2*max_len, pos_dim) if pos_dim > 0 else None

        in_dim = emb_dim
        in_dim += pos_dim * 2
        self.local_encoder = EncoderLSTM(in_dim, hs // 2, 1, True, True, rdp, False, True)
        
        self.in_drop = nn.Dropout(indp)
        self.rnn_drop = nn.Dropout(rdp)
    
    def forward(self, sample, doc_outputs, embs, head_mask, tail_mask):
        local_len = sample['local_len']
        
        local_words_id, local_ners_id, local_coref_id = sample['local_words_id'], sample[
            'local_ners_id'], sample['local_coref_id']
        local_head_mask, local_tail_mask = sample['local_head_mask'], sample[
            'local_tail_mask']  # (path_num, max_path_len)
        local_head_dis, local_tail_dis = sample['local_first_head'] + self.max_len, sample['local_first_tail'] + self.max_len
        
        local_embs = [self.word_embed(local_words_id), self.ner_embed(local_ners_id),
                        self.coref_embed(local_coref_id)]
        if self.head_pos_embed is not None:
            local_embs += [self.head_pos_embed(local_head_dis), self.tail_pos_embed(local_tail_dis)]
        local_embs = torch.cat(local_embs, dim=2)
        local_embs = self.in_drop(local_embs)

        local_outputs, _ = self.local_encoder(local_embs, local_len)
        local_outputs = self.rnn_drop(local_outputs)
        local_head_rep = torch.matmul(local_head_mask.unsqueeze(1), local_outputs)
        local_tail_rep = torch.matmul(local_tail_mask.unsqueeze(1), local_outputs)
    
        return local_head_rep.squeeze(1), local_tail_rep.squeeze(1)
