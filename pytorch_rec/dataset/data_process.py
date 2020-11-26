import pandas as pd
from dataset.utils import *

data=pd.read_csv("short",sep="\t")
for i in range(20):
    data=min_item_inter(data,min_number=5)
    data=min_user_inter(data,min_number=5)
data=unique_id(data)
print(data)
a,b,c=data_for_sasrec(data,max_length=10)
train,pot,neg=a
valid,pov=b
mm,nn=c
print(mm.shape)
for x,y in zip(train,valid):
    print(x,">>>>>>>>>>",y)
    print("kkkkkkkkkkl")
# print(data)