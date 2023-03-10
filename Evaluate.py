'''
评估
'''

from bert4torch.snippets import Callback

import config
from dataset import vaild_dataloader
from tqdm import tqdm
from model import model



def trans_entity2tuple(scores):
    '''
    把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    :param scores:
    :return:
    '''
    batch_entity_ids = set()
    for i,samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(samp):
            label_tag = config.categories_id2label[item.item()]
            if label_tag.startswith('B-'):
                entity_ids.append([i,j,j,label_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (len(entity_ids[-1]) > 0 ) and label_tag.startswith('I-') and (label_tag[2:] == entity_ids[-1][-1]):
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])
        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids



def evaluator(data):
    X,Y,Z = 1e-10,1e-10,1e-10
    X2, Y2, Z2 = 1e-10, 1e-10, 1e-10
    for token_ids,label in tqdm(data):
        scores = model.predict(token_ids) #[btz, seq_len]
        attention_mask = label.gt(0)

        #token粒度
        X += (scores.eq(label) * attention_mask).sum().item()
        Y += scores.gt(0).sum().item()
        Z += label.gt(0).sum().item()

        #entity粒度
        entity_pred = trans_entity2tuple(scores)
        entity_true = trans_entity2tuple(label)
        X2 += len(entity_pred.intersection(entity_true))
        Y2 += len(entity_pred)
        Z2 += len(entity_true)

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2 / Y2, X2 / Z2
    return f1, precision, recall, f2, precision2, recall2



class Evaluator(Callback):
    def __init__(self):
         self.best_val_1 = 0.


    def on_epoch_end(self, steps, epoch, logs=None):
        f1,precision,recall,f2,precision2,recall2 = evaluator(vaild_dataloader)
        if f2 > self.best_val_1:
            self.best_val_1 = f2
            #model.save_weights('best_model.pt')
        print(f'[val-token  level] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}')
        print(f'[val-entity level] f1: {f2:.5f}, p: {precision2:.5f} r: {recall2:.5f} best_f1: {self.best_val_1:.5f}\n')

