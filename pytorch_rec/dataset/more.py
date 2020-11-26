import pandas as pd
from dataset.utils import *

data=pd.read_csv("ml-1m",sep="\t")
for i in range(20):
    data=min_item_inter(data,min_number=5)
    data=min_user_inter(data,min_number=5)
data=unique_id(data)
# data=get_last_max_length_inter(data,max_length=10)
data.to_csv("ml-1m.inter", sep="\t", index=False)