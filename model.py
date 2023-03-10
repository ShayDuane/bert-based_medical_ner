'''
定义bert上的模型
'''
import torch
from bert4torch.models import build_transformer_model,BaseModel
from bert4torch.layers import CRF

import config

import torch.nn as nn

device = 'cpu' if torch.cuda.is_available() else 'cpu'


class Model(BaseModel):

    def __init__(self):
        super(Model,self).__init__()
        self.bert = build_transformer_model(config.config_path,checkpoint_path=config.checkpoint_path,segment_vocab_size=0)
        #segment_vocab_size: int, type_token_ids数量, 默认为2, 如不传入segment_ids则需设置为0
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768,len(config.categories))
        self.crf = CRF(len(config.categories))

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids]) #形状为 [btz, seq_len, hdsz]
        score = self.fc(sequence_output) #形状为 [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()
        return score,attention_mask

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            score,attention_mask = self.forward(token_ids)
            best_path = self.crf.decode(score,attention_mask)
        return best_path


model = Model().to(device)






