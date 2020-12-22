import pickle
import pandas
from pathlib import Path

Link = Path().resolve()
with open(str(Link)+"/data/docs2v.pickle","rb") as files:
    doc2vec=pickle.load(files)
with open(str(Link)+"/data/postid.pickle","rb") as files:
    post_id=pickle.load(files)
#doc2vec = doc2vec.merge(post_id,on = )
doc2vec = doc2vec.rename(columns={"id":"post_id"})
print(post_id.head())
print(doc2vec.head())
doc2vec = doc2vec.merge(post_id,on ="post_id", how = "inner")
doc2vec  = doc2vec[["idp","vector"]]
doc2vec = doc2vec.sort_values(by=["idp"])
doc2vec = list(doc2vec["vector"])
with open(str(Link)+"/data/docs2vec.pickle","wb") as files:
    pickle.dump(doc2vec, files)
print(doc2vec[3])

