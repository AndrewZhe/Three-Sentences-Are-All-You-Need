#!/usr/bin/env bash

cd ../

data_dir='100_lower-doc-new-path'
doc_mode=2

topn=10
hs=256

bz=1
lr=0.001

indp=0.5
rdp=0.2
mlp_dp=0.2
l2=1e-2

classifier='MLP4'
mlp_layer=2

label=97

ner_dim=20
coref_dim=20
pos_dim=20

# Local Encoder
global_encoder='None'
local_encoder='path_bilstm'
doc_embed_required=0
local_required=1
topic_entity_required=0


# Topic Classifier
topic_only=0
topic_rate=0
topic_type="None"
topic_sim="None"
num_topic=5
add_topic_rep=0
rel_type='None'

interact_mode='None'

d_seed=31
m_seed=32

config=${data_dir}/local_path/doc_${doc_embed_required}-pos_${pos_dim}-seed_${d_seed}_${m_seed}
# config=test

python main.py \
    --gpu_no 0 --doc_mode ${doc_mode} --dataset_seed ${d_seed} --model_seed ${m_seed} --mode train --weight_decay ${l2} \
    --lr ${lr} --topn ${topn} --hidden_size $hs --ner_dim ${ner_dim} --coref_dim ${coref_dim} --pos_dim ${pos_dim} \
    --data_dir ../dataset/$data_dir --log_path ../log/${config} --model_path ../model/${config} \
    --pre_trained_embed ../dataset/DocRED_baseline_metadata/glove_100_lower_vec.npy \
    --max_epoch 1000 --warmup_step 0 --eval_step -1 --save_model 0 \
    --input_dropout ${indp} --rnn_dropout ${rdp} --mlp_dropout ${mlp_dp} \
    --doc_embed_required ${doc_embed_required} --local_required ${local_required}

