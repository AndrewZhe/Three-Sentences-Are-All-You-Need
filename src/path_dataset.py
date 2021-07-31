import numpy as np
import json
import random
import math
from collections import defaultdict
import copy
import torch
import os


class Dataset(object):
    def __init__(self, data_file_path, label_num, shuffle, max_sent_len=512,
                 max_entity_pair_num=180000, args=None):
        """
        :param data_type: if train, random dataset
        :param data_file_path: data dict
        :param batch_size:
        :param label_num:
        :param max_sent_len: 512 default
        :param negative_throw_prob: As negative instance are much more than positive instance, we don't use all negative
                instance during training.
        :param shuffle_level: when shuffle dataset, whether to keep all instance in the same document adjacent
        """
        self.label_num = label_num
        self.max_sent_len = max_sent_len
        self.shuffle = shuffle
        self.args = args
        
        self.documents = self.build_dataset(data_file_path)
        
        self.max_entity_pair_num = max_entity_pair_num
        
        self.ptr = 0
        self.size = len(self.documents)
    
    def build_dataset(self, data_file_path):
        def pre_ins(ins):
            for key in ['head_mask', 'tail_mask']:
                new_dt = {}
                for k, v in ins[key].items():
                    new_dt[int(k)] = v
                ins[key] = new_dt
                arr = np.zeros([doc['doc_len']], dtype=np.float32)
                arr_k = list(new_dt.keys())
                arr_v = [new_dt[k] for k in arr_k]
                arr[arr_k] = arr_v
                ins[key] = arr.tolist()
            
            ins_label = [0] * 97
            for label in ins['label']:
                ins_label[label] = 1
            if len(ins['label']) == 0:
                ins_label[0] = 1
            ins['label'] = np.array(ins_label, dtype=np.float32)
            return ins
        
        with open(data_file_path + '.json', 'r') as fh:
            documents = json.load(fh)

        for doc in documents:
            doc['doc_label'] = np.zeros([97], dtype=np.float32)
            doc['sent_doc_mp'] = np.array(doc['sent_doc_mp'], dtype=np.float32)
            for idx, ins in enumerate(doc['pos_ins']):
                doc['pos_ins'][idx] = pre_ins(ins)
                doc['doc_label'] += doc['pos_ins'][idx]['label']
            for idx, ins in enumerate(doc['neg_ins']):
                doc['neg_ins'][idx] = pre_ins(ins)
        
        if self.shuffle:
            random.shuffle(documents)
        
        return documents
    
    def __len__(self):
        return len(self.documents)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        def padding(lst, padding_len):
            data_len = len(lst)
            if data_len > padding_len:
                return np.array(lst[:padding_len])
            elif data_len < padding_len:
                return np.array(lst + [0] * (padding_len - data_len))
            else:
                return np.array(lst)
        
        if self.ptr == self.size:
            self.ptr = 0
            if self.shuffle:
                random.shuffle(self.documents)
            raise StopIteration
        
        doc = self.documents[self.ptr]
        self.ptr += 1
        
        sample = {'support_set': np.array(0), 'sent_doc_mp': np.array(0), 'doc_label': np.array(0)}
        for key in ['doc_len', 'words_id', 'ners_id', 'coref_id', 'sent_num']:
            sample[key] = np.array([doc[key]])  # (1, ?)
        for key in ['head_mask', 'tail_mask', 'label', 'local_len', 'local_head_mask', 'local_tail_mask',
                    'local_words_id', 'local_ners_id', 'local_coref_id', 'local_first_head', 'local_first_tail']:
            sample[key] = []
        
        ins_num = 0
        path_num = []
        same_set_keys = ['head_entity', 'tail_entity', 'head_ner', 'tail_ner', 'pair_ner']
        same_set = {}
        for key in same_set_keys:
            same_set[key] = defaultdict(list)


        if len(doc['pos_ins']) + len(doc['neg_ins']) > self.max_entity_pair_num:
            random.shuffle(doc['neg_ins'])
        for ins_idx, ins in enumerate(doc['pos_ins'] + doc['neg_ins'][:self.max_entity_pair_num - len(doc['pos_ins'])]):
            for key in ['head_mask', 'tail_mask', 'label']:
                sample[key].append(ins[key])
            
            sample['local_len'] += ins['local_len']
            
            for key in ['local_head_mask', 'local_tail_mask']:
                arr = np.zeros([len(ins[key]['k']), doc['max_path_len']], dtype=np.float32)
                arr_k_0 = [idx for idx, mask in enumerate(ins[key]['k']) for m in mask]
                arr_k_1 = [m for idx, mask in enumerate(ins[key]['k']) for m in mask]
                arr_v = [v for vals in ins[key]['v'] for v in vals]
                arr[arr_k_0, arr_k_1] = arr_v
                sample[key] += arr.tolist()

            for key in ['local_first_head', 'local_first_tail']:
                sample[key] += [list(range(-s, 0)) + [0] * (e-s) + list(range(e, doc['max_path_len'])) for s, e in ins[key]]

            
            for key in ['words_id', 'ners_id', 'coref_id']:
                arr = [padding([t for sent_id in sent_ids for t in doc['sent_{}'.format(key)][sent_id]],
                               doc['max_path_len']) for sent_ids in ins['support_set']]
                sample['local_{}'.format(key)] += arr
            
            ins_num += 1
            path_num.append(len(ins['local_len']))
        
        for key in ['head_mask', 'tail_mask', 'local_head_mask', 'local_tail_mask']:
            sample[key] = np.array(sample[key], dtype=np.float32)  # .astype(np.float32)
        
        for key in ['local_words_id', 'local_ners_id', 'local_coref_id']:
            sample[key] = np.array(sample[key], dtype=np.int64)
        
        sample['local_len'] = np.array(sample['local_len'][:sum(path_num)], np.int64)
        sample['label'] = np.array(sample['label'][:ins_num])
        path2ins = np.zeros([sum(path_num), ins_num], np.float32)
        offset = 0
        for pidx, pn in enumerate(path_num):
            arr_k_0 = [offset + t for t in range(pn)]
            arr_k_1 = [pidx] * pn
            path2ins[arr_k_0, arr_k_1] = 1.
            offset += pn
        sample['path2ins'] = path2ins
        sample['doc_label'] = np.array([doc['doc_label']], dtype=np.float32)

        for key in sample:
            sample[key] = torch.tensor(sample[key])
        return sample

