from dataset.utils import *
import pandas as pd
import numpy as np


"""
the data for repeat_net
1.leave_one_out
2.the target is in the session before
"""
def leave_one_out(data,max_seq_length=50):
    all_data=[]

    user_item={}
    data_values=data.values
    for idx,value in enumerate(data_values):
        user_id=value[0]
        item_id=value[1]
        if user_id in user_item.keys():
            user_item[user_id]+=[item_id]
        else:
            user_item[user_id]=[item_id]
    train_data,validation_data,test_data=[],[],[]
    for key,value in user_item.items():
        test=value
        if len(test)>max_seq_length:
            test=test[-max_seq_length:]
        test_data.append(test)
        validation=value[:-1]
        if len(validation)>max_seq_length:
            validation=validation[-max_seq_length:]
        validation_data.append(validation)
        for i in range(2,len(value)-2):
            train=value[:i]
            if len(train)>max_seq_length:
                train=train[-max_seq_length:]
            train_data.append(train)
    all_data.append(train_data)
    all_data.append(validation_data)
    all_data.append(test_data)
    return all_data


def pad_sequences(data,max_length=50):
    for idx,data_value in enumerate(data):
        new_data=np.zeros(shape=(max_length))
        if len(new_data)>len(data_value):
            new_data[-len(data_value):]=data_value
        else:
            new_data=data_value[-max_length-1:]
        data[idx]=new_data
    return data


def read_data_for_repeat_net(data_name="ml-1m",max_seq_length=10):
    data=pd.read_csv(data_name+".inter",sep="\t")
    # for x in range(20):
    #     data=min_user_inter(data,2)
    #     data=min_item_inter(data,5)
    data=unique_id(data)
    # header = ["session_id:token", "item_id:token", "rating:float", "timestamp:float"]
    # data.to_csv("diginetica", sep="\t", index=False, header=header)

# read_data_for_repeat_net(data_name="diginetica")
    train,valid,test=leave_one_out(data,max_seq_length)
    target_in_session=[]
    for x in train:
        target=x[-1]
        residual=x[:-1]
        if target in residual:
            target_in_session.append(1)
            x.append(1)
        else:
            target_in_session.append(0)
            x.append(0)
    train=pad_sequences(train,max_length=max_seq_length+1)
    valid=pad_sequences(valid,max_seq_length)
    test=pad_sequences(test,max_seq_length)
    train=np.array(train)
    valid=np.array(valid)
    test=np.array(test)

    return train,valid,test


