from dataset.utils import *
from mertic import *
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from abstract_model import abstract_model
from Dataset import DataSet


class AP(abstract_model):
    """
    Attention Preference
    ideal: preference first 还是 sequential first
    """
    def __init__(self,
                 data_name="ml-1m",
                 model_name="AP",
                 embedding_size=32,
                 episodes=10,
                 verbose=1,
                 learning_rate=0.001,
                 hidden_size=32,
                 seq_len=6,
                 batch_size=512,
                 min_user_inter=3):
        super(AP, self).__init__(data_name=data_name,model_name=model_name,min_user_number=min_user_inter)
        self.embedding_size=embedding_size
        self.episodes=episodes
        self.verbose=verbose
        self.learning_rate=learning_rate
        self.hidden_size=hidden_size
        self.seq_len=seq_len
        self.batch_size=batch_size

        self.dataset=Data_for_AP(seq_len=self.seq_len)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number
        self.criterion=nn.CrossEntropyLoss()

        self.build_variables()


    def build_variables(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.gru=nn.GRU(input_size=self.embedding_size,hidden_size=self.hidden_size,batch_first=True)
        self.sequential_preference_weight=Sequential_Preference_weight(hidden_size=self.hidden_size)
        self.attention=Attention(hidden_size=self.hidden_size)


    def forward(self,data):
        """

        :param data: [seq_item, user]
        seq_item: batch_size * seq_len
        user: batch_size
        :return:
        """
        seq_item,user=data
        # get_mask
        mask=seq_item.data.eq(0)

        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        user_embedding=self.user_matrix(user)

        all_memory,last_memory=self.gru(seq_item_embedding)
        last_memory=last_memory.transpose(0,1)

        item_embedding=self.attention(all_memory,last_memory,mask)

        user_item_weight=self.sequential_preference_weight(user_embedding=user_embedding,item_embedding=item_embedding)

        user_item_embedding=torch.mul(user_item_weight[:,:1].repeat(1,self.hidden_size),user_embedding)+torch.mul(user_item_weight[:,1:].repeat(1,self.hidden_size),item_embedding)

        return user_item_embedding


    def get_item_matrix_transpose(self):

        return self.item_matrix.weight.t()


    def calculate_loss(self,data):
        seq_item,user,target=data
        user_item_embedding=self.forward([seq_item,user])
        prediction=torch.matmul(user_item_embedding,self.get_item_matrix_transpose())
        loss=self.criterion(prediction,target)

        return loss


    def prediction(self,data):
        seq_item,user=data
        user_item_embedding=self.forward([seq_item,user])
        prediction=torch.matmul(user_item_embedding,self.get_item_matrix_transpose())

        return prediction


class Attention(nn.Module):

    def __init__(self,hidden_size=32):
        super(Attention, self).__init__()
        self.hidden_size=hidden_size
        self.W1=nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.W2=nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.W3=nn.Linear(self.hidden_size,1)
        self.tanh=nn.Tanh()


    def forward(self,all_memory,last_memory,mask=None):
        """

        :param all_memory: batch_size * seq_len * hidden_size
            == W1 ==>> batch_size * seq_len * hidden_size
        :param last_memory: batch_size * 1 * hidden_size
            == W2 ==>> batch_size * 1 * hidden_size
            == repeat ==>> batch_size * seq_len * hidden_size
        ***
        all_memory + last_memory: batch_size * seq_len * hidden_size
            == W3 ==>> batch_size * seq_len * 1
            == squeeze dim 2 ==>> batch_size * seq_len
        ***
        fill the mask: batch_size * seq_len
            == softmax ==>> batch_size * seq_len
            == unsqueeze and repeat ==>> batch_size * seq_len * embedding_size
        ***
        attn * all_memory_value ==>> batch_size * seq_len * embedding_size
            == sum dim 1 ==>> batch_size * embedding_size
        :param mask: batch_size * seq_len
        :return:
        """
        batch_size,seq_len,hidden_size=all_memory.size()
        all_memory_values=all_memory

        all_memory=self.W1(all_memory)
        last_memory=self.W2(last_memory).repeat(1,seq_len,1)

        output=self.tanh(all_memory+last_memory)
        output=self.W3(output).squeeze(2)

        if mask is not None:
            output.masked_fill_(mask,-1e9)

        attn=nn.Softmax(dim=1)(output).unsqueeze(2).repeat(1,1,hidden_size)

        attn_value=torch.mul(attn,all_memory_values).sum(dim=1)

        return attn_value


class Sequential_Preference_weight(nn.Module):

    def __init__(self,hidden_size=32):
        super(Sequential_Preference_weight, self).__init__()
        self.hidden_size=hidden_size
        self.relu=nn.ReLU()
        self.W1=nn.Linear(self.hidden_size,self.hidden_size)
        self.W2=nn.Linear(self.hidden_size,self.hidden_size)
        self.W3=nn.Linear(2*self.hidden_size,self.hidden_size)
        self.W4=nn.Linear(self.hidden_size,2)


    def forward(self,user_embedding,item_embedding):
        """

        :param user_embedding: batch_size * embedding_size
            == W1 ==>> batch_size * embedding_size
        :param item_embedding: batch_size * embedding_size
            ==  W2 ==>> batch_size * embedding_size
        ***
            user_embedding+item_embedding: batch_size * embedding_size
            == W3 ==>> batch_size * 2
            == softmax dim 1 ==>> bathc_size * 2
        :return:
        """
        user_embedding=self.relu(self.W1(user_embedding))
        item_embedding=self.relu(self.W2(item_embedding))
        user_item_embedding=torch.cat([user_embedding,item_embedding],dim=1)
        attn=self.relu(self.W3(user_item_embedding))
        attn=self.W4(attn)

        attn=nn.Softmax(dim=1)(attn)

        return attn


class Data_for_AP(DataSet):

    def __init__(self,seq_len,min_user_number=3,data_name="diginetica"):
        super(Data_for_AP, self).__init__(min_user_number=min_user_number,data_name=data_name)
        self.seq_len=seq_len
        self.data_name=data_name


    def get_data_for_model(self):
        data=self.get_data()

        return data

    def get_data(self):
        data_values=self.data.values

        user_item={}
        for value in data_values:
            user_id=value[0]
            item_id=value[1]
            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]

        train_data,validation_data,test_data=[],[],[]
        for user_id,item_list in user_item.items():
            length=len(item_list)

            test=item_list[-self.seq_len-1:]
            test+=[user_id]
            test_data.append(test)

            valid=item_list[-self.seq_len-2:-1]
            valid+=[user_id]
            validation_data.append(valid)

            for i in range(2,length-2):
                train=item_list[:i]
                train+=[user_id]
                train_data.append(train)

        train_data=self.pad_sequences(train_data,max_length=self.seq_len+2)
        validation_data=self.pad_sequences(validation_data,max_length=self.seq_len+2)
        test_data=self.pad_sequences(test_data,max_length=self.seq_len+2)

        train_data,validation_data,test_data=np.array(train_data),np.array(validation_data),np.array(test_data)

        return train_data,validation_data,test_data


class trainer():
    def __init__(self,model):
        self.model=model
    def train_epoch(self,data):
        train_data_value = data
        total_loss=0
        count=1
        for train_data in get_batch(train_data_value,batch_size=self.model.batch_size):
            """
            train_data: (seq_item,user_id,pos_item,neg_item)
            """
            self.optimizer.zero_grad()
            seq_item = train_data[:, :self.model.seq_len]
            target=train_data[:,self.model.seq_len]
            user_id = train_data[:, self.model.seq_len+1]
            seq_item,target,user_id=torch.LongTensor(seq_item),torch.LongTensor(target),torch.LongTensor(user_id)
            loss = self.model.calculate_loss(data=[seq_item,user_id,target])
            if count%500==0:
                print("the %d step  the current loss is %f"%(count,loss))
            count+=1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss


    def train(self):
        self.model.logging()
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.model.learning_rate)
        train_data,validation_data,test_data=self.model.dataset.get_data_for_model()
        for episode in range(self.model.episodes):
            loss=self.train_epoch(data=train_data)
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                validation=validation_data
                label = validation[:, self.model.seq_len]
                validation=torch.LongTensor(validation)
                seq_item = validation[:, :self.model.seq_len]
                user_id=validation[:,self.model.seq_len+1]
                scores=self.model.prediction([seq_item,user_id])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20,50])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])


model=AP(data_name="diginetica")
trainer=trainer(model=model)
trainer.train()
