import argparse
import math
import torch
import os
import logging
import json
import numpy as np
import random
from model.docBilstm import DocClassifier
from evaluator import Evaluator


def arg_parser():
    parser = argparse.ArgumentParser(description='')
    
    # Random Seed
    parser.add_argument('--dataset_seed', type=int, help="Dataset random seed")
    parser.add_argument('--model_seed', type=int, help="Model random seed")
    
    # Args for utils
    parser.add_argument('--mode', type=str, help="Train or eval")
    parser.add_argument('--doc_mode', type=int, help="Train or eval", default=1)
    parser.add_argument('--gpu_no', type=str, help="GPU number")
    parser.add_argument('--data_dir', type=str, help="Dataset directory")
    
    parser.add_argument('--log_path', type=str, help="Path of log file")
    parser.add_argument('--model_path', type=str, help="Path to store model")
    parser.add_argument('--batch_size', type=int, help="Batch size", default=1)
    parser.add_argument('--max_epoch', type=int, help="Maximum epoch to train")
    parser.add_argument('--warmup_step', type=int, help="Steps before evaluation begins")
    parser.add_argument('--eval_step', type=int, help="Evaluation per step. If -1, eval per epoch")
    parser.add_argument('--save_model', type=int, help="1 to save model, 0 not to save")
    
    # Args for dataset
    parser.add_argument('--max_entity_pair_num', type=int, help="Model random seed", default=1800)

    # Embds
    parser.add_argument('--topn', type=int, help="finetune top n word vec ")
    parser.add_argument('--hidden_size', type=int, help="Size of hidden layer")
    parser.add_argument('--pre_trained_embed', type=str, help="Pre-trained embedding matrix path")
    parser.add_argument('--ner_num', type=int, help="Ner Num", default=7)
    parser.add_argument('--ner_dim', type=int, help="Ner embed size")
    parser.add_argument('--coref_dim', type=int, help="Coref embedd size")
    parser.add_argument('--pos_dim', type=int, help="relative position embed size", default=0)

    # Optimizer (Only useful for Bert)
    parser.add_argument('--lr', type=float, default=1e-3, help='Applies to sgd and adagrad.')
    parser.add_argument('--weight_decay', type=float, default=0, help='.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    
    # Dropout
    parser.add_argument('--input_dropout', type=float, help="")
    parser.add_argument('--rnn_dropout', type=float, help="")
    parser.add_argument('--mlp_dropout', type=float, help="")

    # Classifier
    parser.add_argument('--doc_embed_required', type=int, help="Whether use global entity rep")
    parser.add_argument('--local_required', type=int, help="Whether use local entity rep")

    parser.add_argument('--max_sent_len', type=int, help="The maximum sequence length", default=512)
    parser.add_argument('--class_num', type=int, help="Number of classification labels", default=97)

    args = parser.parse_args()
    
    return args


def train(model, evaluator, logger, dataset_train, dataset_valid, args):
    model.train()
    criterion = torch.nn.BCELoss().cuda()
    
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)

    train_pred, train_gold, train_hit = 0, 0, 0
    best_step, best_valid_p, best_valid_r, best_valid_f1 = 0, 0, 0, 0
    
    train_loss = []
    
    for ep in range(args.max_epoch):
        for sample in dataset_train:

            label = sample['label'].cuda()
            prob = model(sample)
            
            loss = criterion(prob, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            

            predict = prob.round()
            sub_gold = label[:, 1:]
            sub_pred = predict[:, 1:]
            hit = torch.sum(sub_gold * torch.eq(sub_pred, sub_gold).to(torch.float32)).item()
            pred = torch.sum(sub_pred).item()
            gold = torch.sum(sub_gold).item()
            train_gold, train_pred, train_hit = train_gold + gold, train_pred + pred, train_hit + hit

            train_loss.append(loss.item())
            
        logger.info("============= Global Epoch: {} ============".format(ep))
        train_p = train_hit / max(train_pred, 1)
        train_r = train_hit / max(train_gold, 1)
        train_f1 = 2 * train_p * train_r / max((train_p + train_r), 1)
        logger.info("Train Loss: {:.8f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}".format(
                        np.mean(train_loss), train_p, train_r, train_f1))
        
        if ep >= args.warmup_step:
            model.eval()
            valid_loss, valid_p, valid_r, valid_f1, valid_threshold, prob = evaluator.get_evaluation(
                                                                                                    dataset_valid,
                                                                                                     model, criterion,
                                                                                                     args)
            fmt_str = "Valid Loss: {:.8f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}, Threshold: {:.3f}"
            logger.info(fmt_str.format(valid_loss, valid_p, valid_r, valid_f1, valid_threshold))
            model.train()
            
            scheduler.step(valid_f1)
            
            train_gold, train_pred, train_hit =  0, 0, 0
            if best_valid_f1 < valid_f1:
                best_step, best_valid_p, best_valid_r, best_valid_f1 = ep, valid_p, valid_r, valid_f1
                logger.info("Cur Best")
                if args.save_model:
                    # with open(os.path.join(args.model_path, 'best-prob'), 'w') as fh:
                        # json.dump(prob, fh, indent=2)
                    torch.save(model.state_dict(), os.path.join(args.model_path, 'best-model'))
        else:
            # acc_hit, acc_tot = 0, 0
            train_gold, train_pred, train_hit = 0, 0, 0
    logger.info("Best step: {}".format(best_step))
    logger.info("Best Valid P: {:.5f}, R: {:.5f}, F1: {:.5f}".format(best_valid_p, best_valid_r, best_valid_f1))


def eval(model, evaluator, dataset_valid, args):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    valid_loss, valid_p, valid_r, valid_f1, valid_threshold, prob = evaluator.get_evaluation(dataset_valid,
                                                                                             model, criterion,
                                                                                             args)
    print (valid_f1, valid_threshold)
    with open(os.path.join(args.model_path, 'test-prob'), 'w') as fh:
        json.dump(prob, fh, indent=2)


def print_configuration_op(logger, args):
    logger.info("My Configuration:")
    logger.info(args)
    print("My Configuration: ")
    print(args)


def main():
    args = arg_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    
    # Random Seed
    random.seed(args.dataset_seed)
    np.random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    
    # Logger setting
    if args.mode == 'train':
        log_dir = '/'.join(args.log_path.split('/')[:-1])
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if os.path.exists(args.log_path):
            args.log_path += 'finetune'
        
        logging.basicConfig(filename=args.log_path,
                            format='[%(asctime)s:%(message)s]',
                            level=logging.INFO,
                            filemode='w',
                            datefmt='%Y-%m-%d %I:%M:%S')
        logger = logging.getLogger()
        if args.save_model and not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)
        
        print_configuration_op(logger, args)
    else:
        logger = None
    
    # Build Model
    pre_trained_embed = np.load(args.pre_trained_embed)
    opt = dict(
        max_len=args.max_sent_len,
        topn=args.topn,
        num_class=args.class_num,
        hidden_dim=args.hidden_size,
        vocab_size=pre_trained_embed.shape[0],
        emb_dim=pre_trained_embed.shape[1],
        ner_num=args.ner_num,
        ner_dim=args.ner_dim,
        pos_dim=args.pos_dim,
        coref_dim=args.coref_dim,
        input_dropout=args.input_dropout,
        rnn_dropout=args.rnn_dropout,
        mlp_dropout=args.mlp_dropout,
    )
    print(opt)
    model = DocClassifier(opt, pre_trained_embed)
    print(model)
    device_num = len(args.gpu_no.split(','))
    device_ids = list(range(device_num))
    if device_num > 1:
        device_ids = args.gpu_no.split(',')
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model.cuda()
    # parameters = list(model.parameters())
    # import ipdb; ipdb.set_trace()

    if os.path.exists(os.path.join(args.model_path, 'best-model')):
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'best-model')))
        print("================Success Load Model !!==================")
    # if args.mode == 'eval':
        # model.load_state_dict(torch.load(os.path.join(args.model_path, 'best-model')))
        # model.cuda()
    
    evaluator = Evaluator(logger)
    
    # Build Dataset
    print("Building Dataset")
    
    from path_dataset import Dataset
    if args.mode == 'train':
        '''
        dataset_train = Dataset(os.path.join(args.data_dir, 'small_train'), 97, True, max_entity_pair_num=args.max_entity_pair_num, args=args)
        dataset_valid = Dataset(os.path.join(args.data_dir, 'small_train'), 97, False, args=args)
        '''
        dataset_train = Dataset(os.path.join(args.data_dir, 'train'), 97, True, max_entity_pair_num=args.max_entity_pair_num, args=args)
        dataset_valid = Dataset(os.path.join(args.data_dir, 'dev'), 97, False, args=args)
        # '''
    elif args.mode == 'eval_dev':
        dataset_test = Dataset(os.path.join(args.data_dir, 'dev'), 97,  False, args=args)
    elif args.mode == 'eval_test':
        dataset_test = Dataset(os.path.join(args.data_dir, 'test'), 97, False, args=args)
        
    print("Finish Building Dataset")
    # Train and Eval
    
    if args.mode == 'train':
        # train(model, evaluator, logger, dataset_valid, dataset_valid, args)
        train(model, evaluator, logger, dataset_train, dataset_valid, args)
    elif args.mode in ['eval_dev', 'eval_test']:
        eval(model, evaluator, dataset_test, args)
    # else:
    # assert False, "mode is {}, which is not train or eval".format(args.mode)


if __name__ == '__main__':
    main()
