import pandas as pd


data=pd.read_csv("user_log_format1.csv",nrows=1000000)
data.to_csv("Tmall.inter",index=False)