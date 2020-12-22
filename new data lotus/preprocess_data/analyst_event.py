import pandas as pd 
import numpy as np


data = pd.read_csv?
data_post_info = pd.read_csv("post_info.csv")
#merged_inner = pd.merge(left=data, right=data_post_info, left_on='userId', right_on='user_id')
#data = pd.merge(left=data[data["eventId"]==4001],right=data_post_info,left_on='postId', right_on='post_id')
print(len(data_post_info))
print(len(set(data["postId"].to_numpy())))