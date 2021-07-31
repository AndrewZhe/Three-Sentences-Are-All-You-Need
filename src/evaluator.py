import torch
import numpy as np


class Evaluator(object):
    def __init__(self, logger):
        self.logger = logger
        self.tot_gold = 12323  # Dev
    
    def get_evaluation(self, dataset_obj, model, criterion, args):
        with torch.no_grad():
            tot_gold = 0
            hit_score = []
            tot_prob = []
            entity_loss_list = []
            
            for sample in dataset_obj:
                label = sample['label'].cuda()
                prob = model(sample)

                entity_loss = 0., 0.

                entity_loss = criterion(prob, label)

                entity_loss = entity_loss.item()
                
                label_cpu = label[:, 1:].cpu().numpy()
                score = prob[:, 1:].cpu().numpy()
                for i in range(label_cpu.shape[0]):
                    for j in range(label_cpu.shape[1]):
                        hit_score.append((label_cpu[i][j], score[i][j]))
                tot_prob.append(prob.cpu().numpy())
                
                entity_loss_list.append(entity_loss)
                
                '''
                na_mask = predict[:, 0].unsqueeze(-1)  # (data_size, 1)
                hit = torch.sum(sub_label * torch.eq(sub_predict, sub_label).to(torch.float32)).item()
                pred = torch.sum(sub_predict).item()
                gold = torch.sum(sub_label).item()
                tot_gold, tot_pred, tot_hit = tot_gold + gold, tot_pred + pred, tot_hit + hit
                # 考虑na_mask
                sub_predict = (1 - na_mask) * sub_predict
                hit2 = torch.sum(sub_label * torch.eq(sub_predict, sub_label).to(torch.float32)).item()
                pred2 = torch.sum(sub_predict).item()
                tot_hit2, tot_pred2 = tot_hit2 + hit2, tot_pred2 + pred2
                '''

            # 我这里没有用tot_grad, 而是用一个设定好的值，是因为做数据集的时候，会扔掉一不分数据：（C+P2+P3）cover不到的
            # 这样回影响最终的recall计算。 其实没有扔了
            tot_gold = self.tot_gold

            loss = np.mean(entity_loss_list)
            
            # 并不是直接拿0.5 当作threshold，而是动态的寻找。
            hit_score.sort(key=lambda x: -x[1])
            precision, recall = [], []
            correct = 0
            for idx, item in enumerate(hit_score):
                correct += item[0]
                precision.append(correct / (idx + 1))
                recall.append(correct / tot_gold)
            
            precision = np.asarray(precision, dtype='float32')
            recall = np.asarray(recall, dtype='float32')
            f1_arr = (2 * precision * recall / (precision + recall + 1e-20))
            f1 = f1_arr.max()
            f1_pos = f1_arr.argmax()
            threshold = hit_score[f1_pos][1]
            p = precision[f1_pos]
            r = recall[f1_pos]
            
            tot_prob = np.concatenate(tot_prob, axis=0).tolist()
        
        return loss, p, r, f1, threshold, tot_prob
