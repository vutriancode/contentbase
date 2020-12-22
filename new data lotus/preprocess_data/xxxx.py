import pickle
import os
import numpy as np
from pathlib import Path

Link = Path(os.path.abspath(__file__)).parent.parent

with open(str(Link) +"/data/datatraintest.pickle","rb") as tt:
    train, validate, test = pickle.load(tt)
train_u=[]
test_u=[]
train_r=[]
test_r=[]
train_p=[]
test_p=[]
for i in train:
    for m in i:
        train_p.append(m[0])
        train_r.append(m[2])
        train_u.append(m[1])
for i in validate:
    for m in i:
        test_p.append(m[0])
        test_r.append(m[2])
        test_u.append(m[1])
with open(str(Link)+"/data/data_train_test.pickle","wb") as files:
    pickle.dump([train_u,train_p,train_r,test_u,test_p,test_r],files)
