import json
import sys

prob_list = json.load(open(sys.argv[1], 'r'))
dataset = json.load(open(sys.argv[2], 'r'))
id2rel = json.load(open(sys.argv[3], 'r'))
threshold = float(sys.argv[5])

data_list = []
for data in dataset:
    data_list += data['pos_ins'] + data['neg_ins']

assert len(data_list) == len(prob_list)

submmit_list = []


# x['title'], x['h_idx'], x['t_idx'], x['r']
for data, prob in zip(data_list, prob_list):
    for idx, score in enumerate(prob[1:]):
        if score > threshold:
            dt = {
                'title': data['doc_title'],
                'h_idx': data['head'],
                't_idx': data['tail'],
                'r': id2rel[str(idx + 1)]
            }
            submmit_list.append(dt)

with open(sys.argv[4], 'w') as fh:
    json.dump(submmit_list, fh, indent=2)


"""
    python generate_submmit.py /home/huangqz/repos/binary_document_re/model/trunc_data_with_na_all_entity_norm/thre_bz_512_drop_0.5_v3/best-prob \
    /home/huangqz/repos/binary_document_re/dataset/trunc_data_with_na_all_entity_norm/dev.json \
    /home/huangqz/repos/binary_document_re/dataset/DocRED_baseline_metadata/id2rel.json \
    /home/huangqz/repos/binary_document_re/model/trunc_data_with_na_all_entity_norm/thre_bz_512_drop_0.5_v3/dev_submmit.json
"""
