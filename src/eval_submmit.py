#!/usr/bin/env python
import sys
import os
import os.path
import json
import pickle

def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


submmit_path = sys.argv[1]
raw_data_dir = sys.argv[2]
truth_dir = sys.argv[3]
output_dir = sys.argv[4]
label2rel_path = sys.argv[5]

with open(label2rel_path, 'r') as fh:
    label2rel = json.load(fh)

intra_inter = pickle.load(open('/home/huangqz/repos/binary_document_re/dataset/DocRED_baseline_metadata/dev_intra_inter.pk', 'rb'))




"""
python eval_submmit.py \
/home/huangqz/repos/binary_document_re/model/trunc_data_with_na_all_entity_norm/thre_bz_512_drop_0.5_v3/dev_submmit.json \
/home/huangqz/repos/binary_document_re/dataset/ \
/home/huangqz/repos/binary_document_re/dataset/DocRED_baseline_metadata/ \
/home/huangqz/repos/binary_document_re/model/trunc_data_with_na_all_entity_norm/thre_bz_512_drop_0.5_v3/
"""

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fact_in_train_annotated = gen_train_facts(raw_data_dir + "/train_annotated.json", truth_dir)
fact_in_train_distant = gen_train_facts(raw_data_dir + "train_distant.json", truth_dir)

output_filename = os.path.join(output_dir, 'scores.txt')
output_file = open(output_filename, 'w')

truth_file = os.path.join(truth_dir, "dev_dev.json")
truth = json.load(open(truth_file))

std = {}
tot_evidences = 0
titleset = set([])

title2vectexSet = {}

intra_gold, intra_pred, intra_hit = 0, 0, 0
inter_gold, inter_pred, inter_hit = 0, 0, 0


for x in truth:
    title = x['title']
    titleset.add(title)

    vertexSet = x['vertexSet']
    title2vectexSet[title] = vertexSet

    for label in x['labels']:
        r = label2rel[str(label['r'])]

        h_idx = label['h']
        t_idx = label['t']
        std[(title, r, h_idx, t_idx)] = set(label['evidence'])
        tot_evidences += len(label['evidence'])

        if intra_inter[(title, h_idx, t_idx)] == 1:
            inter_gold += 1
        else:
            intra_gold += 1

tot_relations = len(std)

submission_answer_file = submmit_path
tmp = json.load(open(submission_answer_file))
tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
submission_answer = [tmp[0]]
for i in range(1, len(tmp)):
    x = tmp[i]
    y = tmp[i-1]
    if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
        submission_answer.append(tmp[i])

correct_re = 0
correct_evidence = 0
pred_evi = 0

correct_in_train_annotated = 0
correct_in_train_distant = 0
titleset2 = set([])
for x in submission_answer:
    title = x['title']
    h_idx = x['h_idx']
    t_idx = x['t_idx']
    r = x['r']
    if intra_inter[(title, h_idx, t_idx)] == 1:
        inter_pred += 1
    else:
        intra_pred += 1
    titleset2.add(title)
    if title not in title2vectexSet:
        continue
    vertexSet = title2vectexSet[title]

    if 'evidence' in x:
        evi = set(x['evidence'])
    else:
        evi = set([])
    pred_evi += len(evi)

    if (title, r, h_idx, t_idx) in std:
        correct_re += 1
        stdevi = std[(title, r, h_idx, t_idx)]
        correct_evidence += len(stdevi & evi)
        in_train_annotated = in_train_distant = False
        for n1 in vertexSet[h_idx]:
            for n2 in vertexSet[t_idx]:
                if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                    in_train_annotated = True
                if (n1['name'], n2['name'], r) in fact_in_train_distant:
                    in_train_distant = True

        if in_train_annotated:
            correct_in_train_annotated += 1
        if in_train_distant:
            correct_in_train_distant += 1
        
        if intra_inter[(title, h_idx, t_idx)] == 1:
            inter_hit += 1
        else:
            intra_hit += 1

# import ipdb; ipdb.set_trace()

re_p = 1.0 * correct_re / len(submission_answer)
re_r = 1.0 * correct_re / tot_relations
if re_p+re_r == 0:
    re_f1 = 0
else:
    re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

evi_p = 1.0 * correct_evidence / pred_evi if pred_evi>0 else 0
evi_r = 1.0 * correct_evidence / tot_evidences
if evi_p+evi_r == 0:
    evi_f1 = 0
else:
    evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

re_p_ignore_train_annotated = 1.0 * (correct_re-correct_in_train_annotated) / (len(submission_answer)-correct_in_train_annotated)
re_p_ignore_train = 1.0 * (correct_re-correct_in_train_distant) / (len(submission_answer)-correct_in_train_distant)

if re_p_ignore_train_annotated+re_r == 0:
    re_f1_ignore_train_annotated = 0
else:
    re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

if re_p_ignore_train+re_r == 0:
    re_f1_ignore_train = 0
else:
    re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)



intra_p = intra_hit / intra_pred
intra_r = intra_hit / intra_gold
intra_f1 = 2 * intra_p * intra_r / (intra_p + intra_r)


inter_p = inter_hit / inter_pred
inter_r = inter_hit / inter_gold
inter_f1 = 2 * inter_p * inter_r / (inter_p + inter_r)

print ('Intra F1: {}, Inter F1: {}'.format(intra_f1, inter_f1))

print ('RE_F1:', re_f1)
print ('Evi_F1:', evi_f1)
print ('RE_ignore_annotated_F1:', re_f1_ignore_train_annotated)
print ('RE_ignore_distant_F1:', re_f1_ignore_train)

output_file.write("RE_F1: %f\n" % re_f1)
output_file.write("Evi_F1: %f\n" % evi_f1)

output_file.write("RE_ignore_annotated_F1: %f\n" % re_f1_ignore_train_annotated)
output_file.write("RE_ignore_distant_F1: %f\n" % re_f1_ignore_train)


output_file.close()
