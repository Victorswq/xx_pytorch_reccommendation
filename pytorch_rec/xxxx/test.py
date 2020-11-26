import pandas as pd
import numpy as np

train_data=pd.read_csv("base_train.csv")
train_data=train_data.drop("id",axis=1)
train_data=train_data.fillna(0)
train_data_value=train_data.values
zero=np.sum((train_data_value[:,-1]==0))
one=np.sum((train_data_value[:,-1]==1))
new=[]

zeros,ones=[],[]
for x in train_data_value:
    if x[-1]==1:
        ones.append(x)
    else:
        zeros.append(x)
ones,zeros=np.array(ones,dtype=np.int),np.array(zeros,dtype=np.int)

epsilon=0.9
i=0
for x in zeros:
    new.append(x)
    index=np.random.choice(one)
    if np.random.uniform()<epsilon or i<one:
        new.append(ones[index])
    else:
        new.append(ones[i])
        i+=1
new_data=pd.DataFrame(new)
new_data.to_csv("new.csv",index=False)

# da=pd.DataFrame(train_data_value)
# print(da)
# for x in train_data_value:
#     if x[-1]==1:
#         print("yews")

# test_data=pd.read_csv("base_test.csv")
# test_data_value=test_data.values
# print(train_data)
# print(test_data_value.shape)
# print(train_data_value.shape)