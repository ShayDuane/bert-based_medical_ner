from model import Model

import torch.optim as optim
from acc import acc
from Evaluate import Evaluator

from dataset import train_dataloader
from Loss import Loss
from model import model



if __name__ == '__main__':
    model.compile(loss=Loss(),optimizer=optim.Adam(model.parameters(),lr=2e-5),metrics=acc)
    evaluator = Evaluator()
    model.fit(train_dataloader=train_dataloader,epochs=20,steps_per_epoch=None,callbacks=[evaluator])