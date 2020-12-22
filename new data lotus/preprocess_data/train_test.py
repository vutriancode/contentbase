import pickle
import pandas
from pathlib import Path

Link = Path().resolve()
with open(str(Link)+"/data/dataconvert.pickle","rb") as files:
    hist=pickle.load(files)
userid = hist["idu"].unique()
train =[]
validate =[]
test =[]
for i in userid:
    print(i)
    m = hist[hist["idu"]==i]
    m = list(m.to_numpy())
    if len(m)<5:
        train=train+m
    else:
        h = int(len(m)*0.5)
        hh = int(len(m)*0.7)
        hhh = len(m)
        train=train+m[:h]
        validate= validate+m[h:hh]
        test = test+m[hh:hhh]
with open(str(Link)+"/data/datatraintest.pickle","wb") as files:
    pickle.dump([train,validate,test],files)