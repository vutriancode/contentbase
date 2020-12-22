import torch
import pickle
import torch 
import torch.nn as nn
import os
import numpy as np
from pathlib import Path

from model import *
from trainning import *


Link = Path(os.path.abspath(__file__)).parent.parent

def load_data():
    with open(str(Link) +"/data/data_train_test.pickle","rb") as tt:
        train_u,train_p,train_r,test_u,test_p,test_r= pickle.load(tt)
    print(len(train_r))
    print(len(test_r))
    test_u,test_p,test_r = test_u[:128000], test_p[:128000], test_r[:128000]


    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_p),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_p),
                                              torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)
    return train_loader, test_loader 
def main(trainset,testset):
    #train_loader, test_loader = load_data()
    epochs = 100
    with open(str(Link) +"/data/docs2vec.pickle","rb") as tt:
        postEncode = pickle.load(tt)
    len_embedding = 50
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = ContentBaseModel(len_embedding,postEncode,device=device)
    training = Training(model)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
    for i in range(epochs):
        expected_rmse, mae = training.test(test_loader,device=device)
        #expected_rmse, mae = 999, 999
        training.train(train_loader,optimizer,i,expected_rmse, mae,device=device)
        torch.save(model.state_dict(), os.path.join(Link,"result"))



if __name__ == "__main__":
    train_loader,test_loader = load_data()
    main(train_loader,test_loader)