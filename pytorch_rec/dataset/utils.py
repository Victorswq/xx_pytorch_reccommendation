"""
some utils for the build of data_set
"""

import pandas as pd
import numpy as np

def unique_id(data):
    columns=data.columns.values
    # read the name of the columns

    # first: process the data of the session
    session_id=columns[0]
    session_values = data[session_id].values
    session_dict = {}
    session_dict_start=1
    new_session_value=np.zeros_like(session_values)
    for idx,session_value in enumerate(session_values):
        if session_value in session_dict.keys():
            new_session_value[idx]=session_dict[session_value]
        else:
            session_dict[session_value]=session_dict_start
            new_session_value[idx]=session_dict[session_value]
            session_dict_start+=1
    print("the length of user_id is ",session_dict_start)
    data[session_id]=new_session_value

    # second: process the data of the item
    item_id=columns[1]
    item_values=data[item_id].values
    item_dict={}
    item_dict_start=1
    new_item_value=np.ones_like(item_values)
    for idx,item_value in enumerate(item_values):
        if item_value in item_dict.keys():
            new_item_value[idx]=item_dict[item_value]
        else:
            item_dict[item_value]=item_dict_start
            new_item_value[idx]=item_dict[item_value]
            item_dict_start+=1
    data[item_id]=new_item_value
    print("the length of  item_id is ",item_dict_start)

    return data


def min_user_inter(data,min_number=5):
    user_item_dict={}
    data_values=data.values
    for idx,value in enumerate(data_values):
        user_id=value[0]
        if user_id in user_item_dict.keys():
            user_item_dict[user_id]+=[idx]
        else:
            user_item_dict[user_id]=[idx]
    drop_index=[]
    for key,value in user_item_dict.items():
        if len(value)<min_number:
            drop_index+=value
    data=data.drop(drop_index,0)
    data=data.reset_index(drop=True)
    return data


def min_item_inter(data,min_number=5):
    data_values=data.values
    item_user_dict={}
    for idx,value in enumerate(data_values):
        item_id=value[1]
        if item_id in item_user_dict.keys():
            item_user_dict[item_id]+=[idx]
        else:
            item_user_dict[item_id]=[idx]
    drop_index=[]
    for key,value in item_user_dict.items():
        if len(value)<min_number:
            drop_index+=value
    data=data.drop(np.unique(drop_index),0)
    data=data.reset_index(drop=True)
    return data


def sort(data, by, ascending=True):
    data.sort_values(by=by, ascending=ascending, inplace=True, ignore_index=True)


def get_last_max_length_inter(data,max_length=50):
    columns = data.columns.values
    data_values=data.values
    sort(data,by=[columns[0], columns[3]], ascending=True)
    user_inter_dict={}
    for idx,value in enumerate(data_values):
        user_id=value[0]
        if user_id in user_inter_dict.keys():
            user_inter_dict[user_id]+=[idx]
        else:
            user_inter_dict[user_id]=[idx]

    drop_index=[]
    for key,value in user_inter_dict.items():
        drop_index+=value[:-max_length]
    data=data.drop(drop_index,0)
    data=data.reset_index(drop=True)
    return data


def leave_one_out(data_name="ml-1m",max_seq_length=50):
    data=pd.read_csv(data_name+".inter",sep="\t")
    data=unique_id(data)
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


def get_item_or_user_number(data_name="ml-1m"):
    data=pd.read_csv(data_name+".inter",sep="\t")
    data_values=data.values
    item_number=np.max(data_values[:,1])
    user_number=np.max(data_values[:,0])
    return user_number,item_number


def pad_sequences(data,max_length=50):
    data_position = []
    for idx,data_value in enumerate(data):
        new_data=np.zeros(shape=(max_length))
        position=[]
        count=0
        if len(new_data)>len(data_value):
            new_data[-len(data_value):]=data_value
        else:
            new_data=data_value[-max_length-1:]
        data[idx]=new_data
        for x in new_data:
            if x!=0:
                count+=1
            position.append(count)
        position.pop(-1)
        data_position.append(position)
    return data,data_position


def get_batch(data,shuffle=False,batch_size=128):
    if shuffle:
        index = [i for i in range(len(data))]
        np.random.shuffle(index)
        data = data[index]
    for start in range(0,len(data),batch_size):
        end=min(start+batch_size,len(data))
        yield data[start:end]


def get_data(data_name="ml-1m",max_length=50):
    data_value=leave_one_out(data_name=data_name)
    new_data=[]
    for data in data_value:
        new_data.append(pad_sequences(data,max_length=max_length))
    return new_data


def data_for_sasrec(data_name="ml-1m",max_length=50):
    data = pd.read_csv(data_name+".inter", sep="\t")
    values=data.values
    max_item=np.max(values[:,1])
    train,valid,test=[],[],[]
    columns = data.columns.values
    data_values = data.values
    sort(data, by=[columns[0], columns[1]], ascending=True)
    user_inter_dict = {}
    for idx, value in enumerate(data_values):
        user_id = value[0]
        item_id=value[1]
        if user_id in user_inter_dict.keys():
            user_inter_dict[user_id] += [item_id]
        else:
            user_inter_dict[user_id] = [item_id]

    for key, value in user_inter_dict.items():
        test.append(value[-max_length-1:])
        valid.append(value[-max_length-2:-1])
        train.append(value[-max_length-3:-2])
    test,test_position=pad_sequences(test,max_length=max_length+1)
    valid,valid_position=pad_sequences(valid,max_length=max_length+1)
    train,train_position=pad_sequences(train,max_length=max_length+1)
    test,test_position=np.array(test),np.array(test_position)
    valid,valid_position=np.array(valid),np.array(valid_position)
    train,train_position=np.array(train),np.array(train_position)

    neg_item=generate_negative_item(train[:,1:],max_item=max_item)
    train_data = [train,train_position,neg_item]
    test_data=[test,test_position]
    valid_data=[valid,valid_position]

    return train_data,valid_data,test_data


def generate_negative_item(data,max_item=6039):
    neg_item=[]
    for value in data:
        neg=[]
        for x in value:
            a=np.random.choice(max_item,1)[0]+1
            while a==x:
                a = np.random.choice(max_item, 1)[0]+1
            neg.append(a)
        neg_item.append(neg)
    return neg_item





# data=get_data()
# for x,y in data:
#     for a,b in zip(x,y):
#         print(a,">>>>>>>>>>",b)
# a,b,c=leave_one_out()
# for x in a:
#     print(x)