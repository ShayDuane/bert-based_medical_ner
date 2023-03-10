'''
目标函数
'''

import torch.nn as nn
from model import model
class Loss(nn.Module):
    def forward(self,outputs,labels):
        return model.crf(*outputs,labels)
