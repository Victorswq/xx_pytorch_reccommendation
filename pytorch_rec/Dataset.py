import pandas as pd
import numpy as np
import random


class DataSet():

    def __init__(self,data_name="ml-1m",min_user_number=5,min_item_number=5,user_id_index=1,time_id_index=6,item_id_index=2,sep=","):
        self.data_name_=data_name+".inter"
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number
        self.user_id_index=user_id_index
        self.time_id_index=time_id_index
        self.item_id_index=item_id_index
        self.sep=sep
        self.data = self.clean_data(min_user_number=self.min_user_number, min_item_number=self.min_item_number,user_id_index=self.user_id_index,time_id_index=self.time_id_index,sep=self.sep)


    def sort(self,data, by, ascending=True):
        data.sort_values(by=by, ascending=ascending, inplace=True, ignore_index=True)


    def clean_data(self,min_user_number=None,min_item_number=None,user_id_index=1,time_id_index=6,sep="\t"):
        data=pd.read_csv(self.data_name_,sep=sep)
        columns = data.columns.values
        print(columns)
        self.sort(data, by=[columns[user_id_index-1], columns[time_id_index-1]], ascending=True)
        for i in range(1):
            if min_user_number is not None:
                data=self.min_user_inter(data,min_user_number)
            if min_item_number is not None:
                data=self.min_item_inter(data,min_item_number)
        data=self.unique_id(data)
        return data


    def get_data_for_model(self):
        raise NotImplementedError


    def pad_sequences(self,data, max_length=10):
        for idx, data_value in enumerate(data):
            new_data = np.zeros(shape=(max_length),dtype=np.int)
            if len(new_data) > len(data_value):
                new_data[-len(data_value):] = data_value
            else:
                new_data = data_value[-max_length:]
            data[idx] = new_data
        data=np.array(data)
        return data


    def min_user_inter(self,data,min_number):
        user_item_dict = {}
        data_values = data.values
        for idx, value in enumerate(data_values):
            user_id = value[self.user_id_index-1]
            if user_id in user_item_dict.keys():
                user_item_dict[user_id] += [idx]
            else:
                user_item_dict[user_id] = [idx]
        drop_index = []
        for key, value in user_item_dict.items():
            if len(value) < min_number:
                drop_index += value
        data = data.drop(drop_index, 0)
        data = data.reset_index(drop=True)
        return data


    def min_item_inter(self,data,min_number):
        item_user_dict={}
        data_values=data.values
        for idx,value in enumerate(data_values):
            item_id=value[self.item_id_index-1]
            if item_id in item_user_dict.keys():
                item_user_dict[item_id]+=[idx]
            else:
                item_user_dict[item_id]=[idx]
        drop_index=[]
        for key,value in item_user_dict.items():
            if len(value)<min_number:
                drop_index+=value
        data=data.drop(drop_index,0)
        data=data.reset_index(drop=True)
        return data


    def unique_id(self,data):
        columns = data.columns.values
        # read the name of the columns

        # first: process the data of the session
        session_id = columns[self.user_id_index-1]
        session_values = data[session_id].values
        session_dict = {}
        session_dict_start = 1
        new_session_value = np.zeros_like(session_values)
        for idx, session_value in enumerate(session_values):
            if session_value in session_dict.keys():
                new_session_value[idx] = session_dict[session_value]
            else:
                session_dict[session_value] = session_dict_start
                new_session_value[idx] = session_dict[session_value]
                session_dict_start += 1
        self.user_number=session_dict_start
        print("the length of user_id is ", session_dict_start)
        data[session_id] = new_session_value

        # second: process the data of the item
        item_id = columns[self.item_id_index-1]
        item_values = data[item_id].values
        item_dict = {}
        item_dict_start = 1
        new_item_value = np.ones_like(item_values)
        for idx, item_value in enumerate(item_values):
            if item_value in item_dict.keys():
                new_item_value[idx] = item_dict[item_value]
            else:
                item_dict[item_value] = item_dict_start
                new_item_value[idx] = item_dict[item_value]
                item_dict_start += 1
        data[item_id] = new_item_value
        print("the length of  item_id is ", item_dict_start)
        self.item_number=item_dict_start

        return data


    def leave_out_out(self,data,max_seq_length=10):
        all_data = []

        user_item = {}
        data_values = data.values
        for idx, value in enumerate(data_values):
            user_id = value[0]
            item_id = value[1]
            if user_id in user_item.keys():
                user_item[user_id] += [item_id]
            else:
                user_item[user_id] = [item_id]
        train_data, validation_data, test_data = [], [], []
        for key, value in user_item.items():
            test = value
            if len(test) > max_seq_length:
                test = test[-max_seq_length:]
            test_data.append(test)
            validation = value[:-1]
            if len(validation)<1:
                continue
            if len(validation) > max_seq_length:
                validation = validation[-max_seq_length:]
            validation_data.append(validation)
            for i in range(2, len(value) - 2):
                train = value[:i]
                if len(train) > max_seq_length:
                    train = train[-max_seq_length:]
                train_data.append(train)
        train_data=self.pad_sequences(train_data,max_seq_length)
        validation_data=self.pad_sequences(validation_data,max_seq_length)
        test_data=self.pad_sequences(test_data,max_seq_length)
        all_data.append(train_data)
        all_data.append(validation_data)
        all_data.append(test_data)

        return all_data


    def split_by_ratio(self,data,ratio=[0.8,0.1,0.1],shuffle=False):
        data_value = data
        length=len(data_value)
        print("the length is ",length)

        if shuffle is True:
            random_index=np.arange(length)
            data_value=data_value[random_index]

        train_data_length=int(length*ratio[0])
        train_data=data_value[:train_data_length]

        valid_data_length=int(length*ratio[1])
        valid_data=data_value[train_data_length:valid_data_length+train_data_length]

        test_data=data_value[valid_data_length:]

        data=[train_data,valid_data,test_data]

        return data


    def the_last_seq_data(self,data,max_seq_data=10,ratio=[0.8,0.1,0.1],neg_number=1):
        """

        :param data: the data to be read
        :param max_seq_data: like this (usr_id,x1,x2,x3,x4,x5,x6,x7) the last 4 is (user_id,x3,x4,x5,x6,x7) the target is x7, and the user_id may be need
        :return:
        """
        data_values=data.values
        user_item={}

        for value in data_values:
            user_id=value[0]
            item_id=value[1]

            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]

        data=[]
        for key,value in user_item.items():
            length=len(value)
            train=value
            if length>max_seq_data:
                train=value[-max_seq_data-1:]
            train=[key]+train
            target=train[-2]
            for i in range(neg_number):
                neg=np.random.choice(self.item_number)
                while neg==target:
                    neg=np.random.choice(self.item_number)
                train+=[neg]
            data.append(train)

        data=np.array(data)
        train_data,validation_data,test_data=self.split_by_ratio(data=data,ratio=ratio)
        train_data=self.pad_sequences(train_data,max_length=max_seq_data+3)
        validation_data=self.pad_sequences(validation_data,max_length=max_seq_data+3)
        test_data=self.pad_sequences(test_data,max_length=max_seq_data+3)

        return [train_data,validation_data,test_data]

# data=DataSet()
# a,b,c=data.leave_out_out(data.data,max_seq_length=2)
#
# for x in a:
#     print(a)