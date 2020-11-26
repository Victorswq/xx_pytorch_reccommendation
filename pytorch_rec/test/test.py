import pandas as pd


data=pd.read_csv("ml-1m.inter",sep="\t")
columns=data.columns.values
print(columns[3])
data=data.sort_values(by=[columns[0],columns[3]],ascending=True,ignore_index=True)
data=data.values
last=data[0][3]
for i in range(100):
    now=data[i][3]
    print((now-last)/60)
    print("user is ", data[i][0], " and the time is ", data[i][3], " the item id is ", data[i][2])
    last=now