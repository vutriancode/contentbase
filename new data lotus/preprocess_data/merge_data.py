"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from pathlib import Path
#print(Path().resolve())


data_event_history = pd.read_csv(str(Link)+"/data/month_activate_user.csv")

data_event_history = data_event_history.sort_values("userId")
data_event_history = data_event_history.rename(columns={"postId": "post_id"})
post_info = pd.read_csv(str(Link)+"/data/post_info.csv",dtype=str)
post_info.sort_values("post_id")
post_info = post_info.dropna(axis=0, how="any")
post = post_info.drop_duplicates(subset=['post_id', 'title'])
post_info =post["post_id"].unique()
post = data_event_history[data_event_history["post_id"].isin(post_info)][["post_id", "userId","eventId"]]
print(post.head())"""
import pickle
from pathlib import Path

Link = Path().resolve()
with open(str(Link)+"/data/datatraintest.pickle","rb") as files:
    train,validate,test=pickle.load(files)
print(test[1])
print(len(validate))
print(len(train))
