'''
dataset
'''
import numpy as np
import torch
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.tokenizers import Tokenizer
from torch.utils.data import DataLoader,Dataset

import config

seed_everything(42)
device = 'cpu' if torch.cuda.is_available() else 'cpu'


class MyDatatset(ListDataset):
    @staticmethod
    def load_data(file_path):
        Data = []
        with open (file_path,encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):
                if not l:
                    continue
                sentence_data = ['']
                for i,c in enumerate(l.split('\n')):
                    char,label = c.split(' ')
                    sentence_data[0] +=char
                    if label[0] == 'B':
                        sentence_data.append([i,i,label[2:]])
                    elif label[0] == 'I':
                        sentence_data[-1][1] = i
                Data.append(sentence_data)
        return Data
#建立分词器
tokenizer = Tokenizer(config.dict_path,do_lower_case=True)

#重写collate_fn函数
def collate_fn(batch):
    batch_token_ids,batch_labels = [], []
    for sentence in batch:
        tokens = tokenizer.tokenize(sentence[0],maxlen=config.maxlen)
        mapping = tokenizer.rematch(sentence[0],tokens)
        start_mapping = {j[0]: i for i,j in enumerate(mapping) if j}
        end_mapping = {j[-1]:i for i,j in enumerate(mapping) if j}
        '''
        这样设计的原因是考虑分词，mapping是一个list，里面每个词、字都是一个小list
        [[], [0], [1], [2], [3], [4], []]
        如果是按分词语割的，那么这个词的最后一个字j[-1]才是最后的label
        '''
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros(len(token_ids))
        for start,end,label in sentence[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start] = config.categories_label2id['B-'+label]
                labels[start+1:end+1] = config.categories_label2id['I-'+label]
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids),dtype=torch.long,device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels),dtype=torch.long,device=device)
    return batch_token_ids,batch_labels

#制作dataloader
train_dataloader = DataLoader(MyDatatset(config.train_data_path),batch_size=config.batch_size,shuffle=True,collate_fn=
                              collate_fn)
vaild_dataloader = DataLoader(MyDatatset(config.vaild_data_path),batch_size=config.batch_size,collate_fn=collate_fn)
